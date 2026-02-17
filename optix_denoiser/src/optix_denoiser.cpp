/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

 //////////////////////////////////////////////////////////////////////////
 /*

  This sample load GLTF scenes and render using RTX (path tracer)

  The path tracer is rendering in multiple G-Buffers, which are used
  for denoising the "result".

  The final image will be tonemapped, either the "result" or the "denoised"
  and it is the tonemmaped Ldr image that is displayed.

  Look for #OPTIX_D, to find what was added to this example to enable the
  OptiX denoiser.

  Note: regarding semaphores, after raytracing a Vulkan semaphore is emitted
		a the Cuda side waits for its result. When Cuda is done, it sends a
		semaphore as well and this one is added to the Application frame
		wait semaphore. Therefore, CPU isn't been blocked and further Vulkan
		commands can be filled, but the last portion of it, won't be executed
		until the Cuda denoiser is finished. (See m_app->addWaitSemaphore())

 */
 //////////////////////////////////////////////////////////////////////////

#include <array>
#include <filesystem>
#include <vulkan/vulkan_core.h>

#define XR_USE_GRAPHICS_API_VULKAN
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>
#include <openxr/openxr_reflection.h>

#define VMA_IMPLEMENTATION
#include "imgui/imgui_camera_widget.h"
#include "imgui/imgui_helper.h"
#include "imgui/imgui_axis.hpp"
#include "nvh/fileoperations.hpp"
#include "nvp/nvpsystem.hpp"
#include "nvvk/dynamicrendering_vk.hpp"
#include "nvvk/gizmos_vk.hpp"
#include "nvvk/raypicker_vk.hpp"
#include "nvvk/renderpasses_vk.hpp"
#include "nvvk/sbtwrapper_vk.hpp"
#include "nvvk/shaders_vk.hpp"
#include "nvvk/images_vk.hpp"
#include "nvvkhl/alloc_vma.hpp"
#include "nvvkhl/application.hpp"
#include "nvvkhl/element_camera.hpp"
#include "nvvkhl/element_gui.hpp"
#include "nvvkhl/gbuffer.hpp"
#include "nvvkhl/gltf_scene_rtx.hpp"
#include "nvvkhl/gltf_scene_vk.hpp"
#include "nvvkhl/hdr_env.hpp"
#include "nvvkhl/hdr_env_dome.hpp"
#include "nvvkhl/pipeline_container.hpp"
#include "nvvkhl/scene_camera.hpp"
#include "nvvkhl/tonemap_postprocess.hpp"

#include "denoiser.hpp"

#include "shaders/device_host.h"
#include "shaders/dh_bindings.h"
#include "_autogen/pathtrace.rchit.h"
#include "_autogen/pathtrace.rgen.h"
#include "_autogen/pathtrace.rmiss.h"
#include "_autogen/pathtrace.rahit.h"
#include "_autogen/gbuffers.rchit.h"
#include "_autogen/gbuffers.rmiss.h"
#include "nvvkhl/element_benchmark_parameters.hpp"
#include <iostream>
#include <stdexcept>
#include <cstring>
#ifdef _WIN32
#include <Windows.h>
#include <cstdlib>

// Minimal setenv() wrapper for MSVC. overwrite != 0 will replace existing value.
static inline int setenv(const char* name, const char* value, int overwrite)
{
	if (!overwrite)
	{
		size_t required = 0;
		getenv_s(&required, nullptr, 0, name);
		if (required) // variable already set
			return 0;
	}
	return _putenv_s(name, value);
}

// usleep in microseconds -> Sleep in milliseconds
static inline void usleep(unsigned int usec)
{
	Sleep((usec + 999) / 1000);
}
#endif

// Add this at the top of your file (after includes, before any functions)
struct OpenXRState
{
	XrInstance instance = XR_NULL_HANDLE;
	XrSystemId systemId = XR_NULL_SYSTEM_ID;
	XrSession session = XR_NULL_HANDLE;
	XrSpace referenceSpace = XR_NULL_HANDLE;
	XrSwapchain swapchain = XR_NULL_HANDLE;
	std::vector<XrSwapchainImageVulkanKHR> swapchainImages;
	std::vector<VkImageLayout> swapchainImageLayouts;
	XrGraphicsBindingVulkan2KHR graphicsBinding{};
	uint32_t swapchainWidth = 0;
	uint32_t swapchainHeight = 0;

	bool isInitialized() const { return instance != XR_NULL_HANDLE; }
	void cleanup()
	{
		if (swapchain != XR_NULL_HANDLE)
		{
			xrDestroySwapchain(swapchain);
			swapchain = XR_NULL_HANDLE;
		}
		if (session != XR_NULL_HANDLE)
		{
			xrDestroySession(session);
			session = XR_NULL_HANDLE;
		}
		if (instance != XR_NULL_HANDLE)
		{
			xrDestroyInstance(instance);
			instance = XR_NULL_HANDLE;
		}
	}
};

OpenXRState g_openXRState;
bool m_enableXR = false;
bool g_useVulkan2 = false;

std::shared_ptr<nvvkhl::ElementCamera> g_elemCamera;
std::shared_ptr<nvvkhl::ElementBenchmarkParameters> g_elemBenchmark;

XrFrameWaitInfo frameWaitInfo = { XR_TYPE_FRAME_WAIT_INFO };
XrFrameState frameState = { XR_TYPE_FRAME_STATE };
std::vector<XrCompositionLayerProjectionView> projectionViews;
// System properties
XrSystemProperties systemProperties = { XR_TYPE_SYSTEM_PROPERTIES };

#ifndef XR_CHECK
#define XR_CHECK(x)                                                    \
  do                                                                   \
  {                                                                    \
    XrResult result = (x);                                             \
    if (XR_FAILED(result))                                             \
    {                                                                  \
      std::cerr << "OpenXR error: " << result                          \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
      throw std::runtime_error("OpenXR call failed");                  \
    }                                                                  \
  } while (0)
#endif



XrBool32 XRAPI_PTR debugCallback(
	XrDebugUtilsMessageSeverityFlagsEXT messageSeverity,
	XrDebugUtilsMessageTypeFlagsEXT messageType,
	const XrDebugUtilsMessengerCallbackDataEXT* callbackData,
	void* userData)
{

	std::cerr << "\n=== OpenXR Validation Message ===" << std::endl;
	std::cerr << "Severity: ";
	switch (messageSeverity)
	{
	case XR_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT:
		std::cerr << "Verbose";
		break;
	case XR_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT:
		std::cerr << "Info";
		break;
	case XR_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
		std::cerr << "Warning";
		break;
	case XR_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
		std::cerr << "Error";
		break;
	}

	std::cerr << "\nType: ";
	switch (messageType)
	{
	case XR_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT:
		std::cerr << "General";
		break;
	case XR_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT:
		std::cerr << "Validation";
		break;
	case XR_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT:
		std::cerr << "Performance";
		break;
	}

	std::cerr << "\nFunction: " << callbackData->functionName;
	std::cerr << "\nMessage: " << callbackData->message << std::endl;

	return XR_FALSE;
}


void setupDebugMessenger()
{
	PFN_xrCreateDebugUtilsMessengerEXT pfnCreateDebugUtilsMessengerEXT = nullptr;
	xrGetInstanceProcAddr(g_openXRState.instance, "xrCreateDebugUtilsMessengerEXT",
		reinterpret_cast<PFN_xrVoidFunction*>(&pfnCreateDebugUtilsMessengerEXT));

	if (!pfnCreateDebugUtilsMessengerEXT)
	{
		std::cout << "Debug utils extension not available" << std::endl;
		return;
	}

	XrDebugUtilsMessengerCreateInfoEXT debugCreateInfo = { XR_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT };
	debugCreateInfo.messageSeverities = XR_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
		XR_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
		XR_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
		XR_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
	debugCreateInfo.messageTypes = XR_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
		XR_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
		XR_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT |
		XR_DEBUG_UTILS_MESSAGE_TYPE_CONFORMANCE_BIT_EXT;
	debugCreateInfo.userCallback = debugCallback;
	debugCreateInfo.userData = nullptr;

	XrDebugUtilsMessengerEXT debugMessenger;
	XrResult result = pfnCreateDebugUtilsMessengerEXT(g_openXRState.instance, &debugCreateInfo, &debugMessenger);
	if (result == XR_SUCCESS)
	{
		std::cout << "Debug messenger created successfully" << std::endl;
	}
	else
	{
		std::cout << "Failed to create debug messenger: " << result << std::endl;
	}
}



void createOpenXRSwapchain()
{
	// Try to get recommended resolution from OpenXR
	uint32_t viewCount = 0;
	xrEnumerateViewConfigurationViews(g_openXRState.instance, g_openXRState.systemId,
		XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, 0, &viewCount, nullptr);

	std::vector<XrViewConfigurationView> views(viewCount, { XR_TYPE_VIEW_CONFIGURATION_VIEW });
	xrEnumerateViewConfigurationViews(g_openXRState.instance, g_openXRState.systemId,
		XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, viewCount, &viewCount, views.data());

	uint32_t width = 0;
	uint32_t height = 0;

	if (viewCount >= 2)
	{
		width = views[0].recommendedImageRectWidth;
		height = views[0].recommendedImageRectHeight;
		std::cout << "OpenXR recommended resolution: " << width << "x" << height << std::endl;
	}
	else
	{
		width = 2016;
		height = 2240;
		std::cout << "Using fallback resolution: " << width << "x" << height << std::endl;
	}

	// Query supported swapchain formats from the runtime
	uint32_t formatCount = 0;
	xrEnumerateSwapchainFormats(g_openXRState.session, 0, &formatCount, nullptr);
	std::vector<int64_t> supportedFormats(formatCount);
	xrEnumerateSwapchainFormats(g_openXRState.session, formatCount, &formatCount, supportedFormats.data());

	std::cout << "Supported swapchain formats (" << formatCount << "):" << std::endl;
	for (auto fmt : supportedFormats)
	{
		std::cout << "  VkFormat: " << fmt << std::endl;
	}

	// Pick the best available format (prefer SRGB, then UNORM)
	int64_t chosenFormat = supportedFormats[0]; // fallback to first supported
	const int64_t preferredFormats[] = {
		VK_FORMAT_R8G8B8A8_SRGB,
		VK_FORMAT_B8G8R8A8_SRGB,
		VK_FORMAT_R8G8B8A8_UNORM,
		VK_FORMAT_B8G8R8A8_UNORM,
	};
	for (auto preferred : preferredFormats)
	{
		for (auto supported : supportedFormats)
		{
			if (supported == preferred)
			{
				chosenFormat = preferred;
				goto formatFound;
			}
		}
	}
formatFound:
	std::cout << "Chosen swapchain format: " << chosenFormat << std::endl;

	XrSwapchainCreateInfo swapchainCreateInfo = { XR_TYPE_SWAPCHAIN_CREATE_INFO };
	swapchainCreateInfo.usageFlags =
		XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT |
		XR_SWAPCHAIN_USAGE_TRANSFER_DST_BIT;
	swapchainCreateInfo.format = chosenFormat;
	swapchainCreateInfo.sampleCount = 1;
	swapchainCreateInfo.width = width;
	swapchainCreateInfo.height = height;
	swapchainCreateInfo.faceCount = 1;
	swapchainCreateInfo.arraySize = 2; // Stereo
	swapchainCreateInfo.mipCount = 1;

	g_openXRState.swapchainWidth = width;
	g_openXRState.swapchainHeight = height;

	XrResult result = xrCreateSwapchain(g_openXRState.session, &swapchainCreateInfo, &g_openXRState.swapchain);
	if (result != XR_SUCCESS)
	{
		std::cerr << "xrCreateSwapchain failed with: " << result << std::endl;
		throw std::runtime_error("Failed to create OpenXR swapchain");
	}

	uint32_t imageCount;
	xrEnumerateSwapchainImages(g_openXRState.swapchain, 0, &imageCount, nullptr);
	g_openXRState.swapchainImages.resize(imageCount, { XR_TYPE_SWAPCHAIN_IMAGE_VULKAN_KHR });
	xrEnumerateSwapchainImages(g_openXRState.swapchain, imageCount, &imageCount,
		reinterpret_cast<XrSwapchainImageBaseHeader*>(g_openXRState.swapchainImages.data()));
	g_openXRState.swapchainImageLayouts.assign(imageCount, VK_IMAGE_LAYOUT_UNDEFINED);

	std::cout << "OpenXR swapchain created with " << imageCount << " images at "
		<< width << "x" << height << std::endl;
}

XrVersion getSupportedOpenXRVersion() {
	// Try different versions from highest to lowest
	XrVersion versions[] = {
		XR_MAKE_VERSION(1, 1, 0),
		XR_MAKE_VERSION(1, 0, 34),
		XR_MAKE_VERSION(1, 0, 33),
		XR_MAKE_VERSION(1, 0, 32),
		XR_MAKE_VERSION(1, 0, 31),
		XR_MAKE_VERSION(1, 0, 30),
		XR_MAKE_VERSION(1, 0, 29),
		XR_MAKE_VERSION(1, 0, 28),
		XR_MAKE_VERSION(1, 0, 27),
		XR_MAKE_VERSION(1, 0, 26),
		XR_MAKE_VERSION(1, 0, 25),
		XR_MAKE_VERSION(1, 0, 24),
		XR_MAKE_VERSION(1, 0, 23),
		XR_MAKE_VERSION(1, 0, 22),
		XR_MAKE_VERSION(1, 0, 21),
		XR_MAKE_VERSION(1, 0, 20),
	};

	XrApplicationInfo testAppInfo{};
	strcpy(testAppInfo.applicationName, "Test");
	strcpy(testAppInfo.engineName, "Test");

	for (XrVersion version : versions) {
		testAppInfo.apiVersion = version;

		XrInstanceCreateInfo testCreateInfo = { XR_TYPE_INSTANCE_CREATE_INFO };
		testCreateInfo.applicationInfo = testAppInfo;
		testCreateInfo.enabledExtensionCount = 0;
		testCreateInfo.enabledExtensionNames = nullptr;
		testCreateInfo.enabledApiLayerCount = 0;

		XrInstance testInstance;
		XrResult result = xrCreateInstance(&testCreateInfo, &testInstance);

		if (result == XR_SUCCESS) {
			std::cout << "Supported OpenXR version: "
				<< XR_VERSION_MAJOR(version) << "."
				<< XR_VERSION_MINOR(version) << "."
				<< XR_VERSION_PATCH(version) << std::endl;
			xrDestroyInstance(testInstance);
			return version;
		}
	}

	std::cout << "Warning: Could not determine supported OpenXR version, using 1.0.0" << std::endl;
	return XR_MAKE_VERSION(1, 0, 0);
}

// Helper: create VkInstance through OpenXR (enable2)
static VkInstance xrCreateVkInstance(XrInstance xrInstance, XrSystemId systemId)
{
	PFN_xrCreateVulkanInstanceKHR pfnCreateInstance = nullptr;
	xrGetInstanceProcAddr(xrInstance, "xrCreateVulkanInstanceKHR",
		reinterpret_cast<PFN_xrVoidFunction*>(&pfnCreateInstance));
	if (!pfnCreateInstance)
		throw std::runtime_error("xrCreateVulkanInstanceKHR not available");

	VkApplicationInfo appInfo{ VK_STRUCTURE_TYPE_APPLICATION_INFO };
	appInfo.pApplicationName = "VK_DENOISE_VR";
	appInfo.pEngineName = "NVVK";
	appInfo.apiVersion = VK_API_VERSION_1_2;

	std::vector<const char*> instExts;
	instExts.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
#ifdef _WIN32
	instExts.push_back(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
#endif
	instExts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

	VkInstanceCreateInfo vkInstInfo{ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
	vkInstInfo.pApplicationInfo = &appInfo;
	vkInstInfo.enabledExtensionCount = static_cast<uint32_t>(instExts.size());
	vkInstInfo.ppEnabledExtensionNames = instExts.data();

	XrVulkanInstanceCreateInfoKHR xrInstInfo{ XR_TYPE_VULKAN_INSTANCE_CREATE_INFO_KHR };
	xrInstInfo.systemId = systemId;
	xrInstInfo.createFlags = 0;
	xrInstInfo.pfnGetInstanceProcAddr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(vkGetInstanceProcAddr);
	xrInstInfo.vulkanCreateInfo = &vkInstInfo;
	xrInstInfo.vulkanAllocator = nullptr;

	VkInstance vkInstance = VK_NULL_HANDLE;
	VkResult vkCreateResult = VK_SUCCESS;
	XrResult xrRes = pfnCreateInstance(xrInstance, &xrInstInfo, &vkInstance, &vkCreateResult);
	if (xrRes != XR_SUCCESS || vkInstance == VK_NULL_HANDLE || vkCreateResult != VK_SUCCESS)
		throw std::runtime_error("xrCreateVulkanInstanceKHR failed");

	return vkInstance;
}

// Helper: create VkDevice through OpenXR (enable2)
// Helper: create VkDevice through OpenXR (enable2)
static VkDevice xrCreateVkDevice(XrInstance xrInstance, XrSystemId systemId, VkPhysicalDevice physDev, uint32_t graphicsQueueFamily)
{
	PFN_xrCreateVulkanDeviceKHR pfnCreateDevice = nullptr;
	xrGetInstanceProcAddr(xrInstance, "xrCreateVulkanDeviceKHR",
		reinterpret_cast<PFN_xrVoidFunction*>(&pfnCreateDevice));
	if (!pfnCreateDevice)
		throw std::runtime_error("xrCreateVulkanDeviceKHR not available");

	float queuePriority = 1.0f;
	VkDeviceQueueCreateInfo qinfo{ VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
	qinfo.queueFamilyIndex = graphicsQueueFamily;
	qinfo.queueCount = 1;
	qinfo.pQueuePriorities = &queuePriority;

	std::vector<const char*> devExts = {
	  VK_KHR_SWAPCHAIN_EXTENSION_NAME,
	  VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
	  VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
	  VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
	  VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
	  VK_KHR_RAY_QUERY_EXTENSION_NAME,
	  VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
	  VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME,
	  VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
	  VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME,         // Required for vkQueueSubmit2
	  VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME,
  #ifdef _WIN32
	  VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
	  VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
	  VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME,
	  VK_KHR_EXTERNAL_FENCE_EXTENSION_NAME,
	  VK_KHR_EXTERNAL_FENCE_WIN32_EXTENSION_NAME,
  #else
	  VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
	  VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
	  VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
	  VK_KHR_EXTERNAL_FENCE_EXTENSION_NAME,
	  VK_KHR_EXTERNAL_FENCE_FD_EXTENSION_NAME,
  #endif
	};

	// Chain all required Vulkan 1.2+ features for ray tracing and sync2
	VkPhysicalDeviceSynchronization2Features sync2Features{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES };
	sync2Features.synchronization2 = VK_TRUE;

	VkPhysicalDeviceRayQueryFeaturesKHR rayQueryFeatures{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR };
	rayQueryFeatures.rayQuery = VK_TRUE;
	rayQueryFeatures.pNext = &sync2Features;

	VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeatures{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR };
	rtPipelineFeatures.rayTracingPipeline = VK_TRUE;
	rtPipelineFeatures.pNext = &rayQueryFeatures;

	VkPhysicalDeviceAccelerationStructureFeaturesKHR accelFeatures{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR };
	accelFeatures.accelerationStructure = VK_TRUE;
	accelFeatures.pNext = &rtPipelineFeatures;

	VkPhysicalDeviceVulkan12Features vulkan12Features{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
	vulkan12Features.bufferDeviceAddress = VK_TRUE;
	vulkan12Features.descriptorIndexing = VK_TRUE;
	vulkan12Features.runtimeDescriptorArray = VK_TRUE;
	vulkan12Features.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
	vulkan12Features.timelineSemaphore = VK_TRUE;  // Required for OptiX interop semaphores
	vulkan12Features.pNext = &accelFeatures;

	VkPhysicalDeviceVulkan11Features vulkan11Features{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES };
	vulkan11Features.pNext = &vulkan12Features;

	VkPhysicalDeviceFeatures2 features2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
	features2.features.samplerAnisotropy = VK_TRUE;
	features2.pNext = &vulkan11Features;

	VkDeviceCreateInfo devInfo{ VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
	devInfo.queueCreateInfoCount = 1;
	devInfo.pQueueCreateInfos = &qinfo;
	devInfo.pEnabledFeatures = nullptr;  // Must be NULL when using pNext feature chain
	devInfo.pNext = &features2;          // Chain features via pNext instead
	devInfo.enabledExtensionCount = static_cast<uint32_t>(devExts.size());
	devInfo.ppEnabledExtensionNames = devExts.data();

	XrVulkanDeviceCreateInfoKHR xrDevInfo{ XR_TYPE_VULKAN_DEVICE_CREATE_INFO_KHR };
	xrDevInfo.systemId = systemId;
	xrDevInfo.createFlags = 0;
	xrDevInfo.pfnGetInstanceProcAddr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(vkGetInstanceProcAddr);
	xrDevInfo.vulkanPhysicalDevice = physDev;
	xrDevInfo.vulkanCreateInfo = &devInfo;
	xrDevInfo.vulkanAllocator = nullptr;

	VkDevice device = VK_NULL_HANDLE;
	VkResult vkCreateResult = VK_SUCCESS;
	XrResult xrRes = pfnCreateDevice(xrInstance, &xrDevInfo, &device, &vkCreateResult);
	if (xrRes != XR_SUCCESS || device == VK_NULL_HANDLE || vkCreateResult != VK_SUCCESS)
	{
		std::cerr << "xrCreateVulkanDeviceKHR failed: XrResult=" << xrRes
			<< " VkResult=" << vkCreateResult << std::endl;
		throw std::runtime_error("xrCreateVulkanDeviceKHR failed");
	}

	return device;
}
// Add this helper function
bool comparePhysicalDevices(VkPhysicalDevice dev1, VkPhysicalDevice dev2)
{
	if (dev1 == VK_NULL_HANDLE || dev2 == VK_NULL_HANDLE)
	{
		std::cout << "Warning: One or both physical devices are null handles" << std::endl;
		return false;
	}

	if (dev1 == dev2)
		return true;

	// Get device properties to compare
	VkPhysicalDeviceProperties prop1, prop2;
	vkGetPhysicalDeviceProperties(dev1, &prop1);
	vkGetPhysicalDeviceProperties(dev2, &prop2);

	// Compare by name and device ID
	return (strcmp(prop1.deviceName, prop2.deviceName) == 0 &&
		prop1.deviceID == prop2.deviceID &&
		prop1.vendorID == prop2.vendorID);
}

static std::vector<std::string> xrGetRequiredVulkanInstanceExts(XrInstance xrInstance, XrSystemId systemId)
{
	uint32_t len = 0;
	PFN_xrGetVulkanInstanceExtensionsKHR pfnInstExts = nullptr;
	xrGetInstanceProcAddr(xrInstance, "xrGetVulkanInstanceExtensionsKHR",
		reinterpret_cast<PFN_xrVoidFunction*>(&pfnInstExts));
	std::vector<std::string> result;
	if (!pfnInstExts) return result;

	XrResult r = pfnInstExts(xrInstance, systemId, 0, &len, nullptr);
	if (r != XR_SUCCESS || len == 0) return result;

	std::string buffer(len, '\0');
	r = pfnInstExts(xrInstance, systemId, len, &len, buffer.data());
	if (r != XR_SUCCESS) return result;

	// Split space-delimited list
	std::istringstream iss(buffer);
	std::string ext;
	while (iss >> ext) result.push_back(ext);
	return result;
}

// Returns space-delimited list of required device extensions from the runtime
static std::vector<std::string> xrGetRequiredVulkanDeviceExts(XrInstance xrInstance, XrSystemId systemId)
{
	uint32_t len = 0;
	PFN_xrGetVulkanDeviceExtensionsKHR pfnDevExts = nullptr;
	xrGetInstanceProcAddr(xrInstance, "xrGetVulkanDeviceExtensionsKHR",
		reinterpret_cast<PFN_xrVoidFunction*>(&pfnDevExts));
	std::vector<std::string> result;
	if (!pfnDevExts) return result;

	XrResult r = pfnDevExts(xrInstance, systemId, 0, &len, nullptr);
	if (r != XR_SUCCESS || len == 0) return result;

	std::string buffer(len, '\0');
	r = pfnDevExts(xrInstance, systemId, len, &len, buffer.data());
	if (r != XR_SUCCESS) return result;

	std::istringstream iss(buffer);
	std::string ext;
	while (iss >> ext) result.push_back(ext);
	return result;
}

// Add near the top (after includes, before initializeOpenXR):
static XrResult xrGetSystemWithRetry(XrInstance instance, XrSystemId* outSystemId, XrFormFactor formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY)
{
	XrSystemGetInfo info{ XR_TYPE_SYSTEM_GET_INFO };
	info.formFactor = formFactor;

	// Retry up to 10 times with 300ms delay (~3s total)
	for (int attempt = 0; attempt < 10; ++attempt)
	{
		XrResult r = xrGetSystem(instance, &info, outSystemId);
		if (r == XR_SUCCESS)
			return r;

		// Common when runtime not ready or HMD not yet in PCVR: XR_ERROR_FORM_FACTOR_UNAVAILABLE (-35)
#ifdef _WIN32
		usleep(300000); // 300 ms
#else
		std::this_thread::sleep_for(std::chrono::milliseconds(300));
#endif
	}
	return XR_ERROR_RUNTIME_FAILURE;
}

void createInstanceOpenXR() {
	std::cout << "--Creating OPENXR instance--" << std::endl;

	XrApplicationInfo appInfo{};
	strcpy(appInfo.applicationName, "VK_DENOISE_VR");
	strcpy(appInfo.engineName, "NVVK");
	appInfo.apiVersion = XR_MAKE_VERSION(1, 0, 34);

	// Query available extensions
	uint32_t extCount = 0;
	xrEnumerateInstanceExtensionProperties(nullptr, 0, &extCount, nullptr);
	std::vector<XrExtensionProperties> extensions(extCount, { XR_TYPE_EXTENSION_PROPERTIES });
	xrEnumerateInstanceExtensionProperties(nullptr, extCount, &extCount, extensions.data());

	bool hasVulkan2 = false;
	bool hasVulkan1 = false;
	for (const auto& ext : extensions) {
		if (strcmp(ext.extensionName, XR_KHR_VULKAN_ENABLE2_EXTENSION_NAME) == 0) hasVulkan2 = true;
		if (strcmp(ext.extensionName, XR_KHR_VULKAN_ENABLE_EXTENSION_NAME) == 0)  hasVulkan1 = true;
	}

	std::vector<const char*> enabledExtensions;
	if (hasVulkan2) {
		enabledExtensions.push_back(XR_KHR_VULKAN_ENABLE2_EXTENSION_NAME);
		g_useVulkan2 = true;
		std::cout << "Enabling extension: " << XR_KHR_VULKAN_ENABLE2_EXTENSION_NAME << std::endl;
	}
	if (hasVulkan1) {
		enabledExtensions.push_back(XR_KHR_VULKAN_ENABLE_EXTENSION_NAME);
		std::cout << "Enabling extension: " << XR_KHR_VULKAN_ENABLE_EXTENSION_NAME << std::endl;
	}
	// As last resort, if neither is present, bail out
	if (!hasVulkan2 && !hasVulkan1) {
		throw std::runtime_error("OpenXR runtime does not support Vulkan enable extensions");
	}

	XrInstanceCreateInfo instanceCreateInfo = { XR_TYPE_INSTANCE_CREATE_INFO };
	instanceCreateInfo.applicationInfo = appInfo;
	instanceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(enabledExtensions.size());
	instanceCreateInfo.enabledExtensionNames = enabledExtensions.data();

	XrResult result = xrCreateInstance(&instanceCreateInfo, &g_openXRState.instance);
	if (result != XR_SUCCESS) {
		std::cerr << "Failed to create OpenXR instance: " << result << std::endl;
		throw std::runtime_error("OpenXR instance creation failed");
	}

	std::cout << "OpenXR instance created successfully" << std::endl;
}

void checkVulkanDeviceForOpenXR(VkPhysicalDevice physicalDevice, VkDevice device, VkInstance instance) {
	std::cout << "\n=== Vulkan Device Diagnostics for OpenXR ===" << std::endl;

	// Check device properties
	VkPhysicalDeviceProperties props;
	vkGetPhysicalDeviceProperties(physicalDevice, &props);
	std::cout << "Device: " << props.deviceName << std::endl;
	std::cout << "Vendor ID: " << props.vendorID << std::endl;
	std::cout << "Device ID: " << props.deviceID << std::endl;
	std::cout << "Device Type: " << props.deviceType << std::endl;
	std::cout << "API Version: "
		<< VK_VERSION_MAJOR(props.apiVersion) << "."
		<< VK_VERSION_MINOR(props.apiVersion) << "."
		<< VK_VERSION_PATCH(props.apiVersion) << std::endl;

	// Check queue families
	uint32_t queueFamilyCount = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
	std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
	vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());

	std::cout << "\nQueue Families (" << queueFamilyCount << "):" << std::endl;
	for (uint32_t i = 0; i < queueFamilyCount; ++i) {
		std::cout << "  [" << i << "] Flags: ";
		if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) std::cout << "GRAPHICS ";
		if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) std::cout << "COMPUTE ";
		if (queueFamilies[i].queueFlags & VK_QUEUE_TRANSFER_BIT) std::cout << "TRANSFER ";
		std::cout << "| Count: " << queueFamilies[i].queueCount << std::endl;
	}

	// Check for required extensions
	uint32_t extensionCount = 0;
	vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, nullptr);
	std::vector<VkExtensionProperties> extensions(extensionCount);
	vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extensionCount, extensions.data());

	std::cout << "\nRequired extensions for SteamVR:" << std::endl;
	const char* requiredExts[] = {
		VK_KHR_SWAPCHAIN_EXTENSION_NAME,
		VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
		VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
		VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME,
		VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME
	};

	for (const char* reqExt : requiredExts) {
		bool found = false;
		for (const auto& ext : extensions) {
			if (strcmp(ext.extensionName, reqExt) == 0) {
				found = true;
				break;
			}
		}
		std::cout << "  " << (found ? "✓" : "✗") << " " << reqExt << std::endl;
	}
}

bool initializeOpenXR(std::shared_ptr<nvvk::Context> context) {
	std::cout << "\n=== Initializing OpenXR ===" << std::endl;
	try {
		if (g_openXRState.instance == XR_NULL_HANDLE) {
			createInstanceOpenXR();
		}

		XrResult result = xrGetSystemWithRetry(g_openXRState.instance, &g_openXRState.systemId);
		if (result != XR_SUCCESS) {
			std::cerr << "Failed to get OpenXR system: " << result << std::endl;
			return false;
		}

		// Query graphics requirements (legacy API works across runtimes)
		PFN_xrGetVulkanGraphicsRequirementsKHR pfnReq = nullptr;
		xrGetInstanceProcAddr(g_openXRState.instance, "xrGetVulkanGraphicsRequirementsKHR",
			reinterpret_cast<PFN_xrVoidFunction*>(&pfnReq));
		if (pfnReq) {
			XrGraphicsRequirementsVulkanKHR req{ XR_TYPE_GRAPHICS_REQUIREMENTS_VULKAN_KHR };
			XR_CHECK(pfnReq(g_openXRState.instance, g_openXRState.systemId, &req));
			std::cout << "Runtime requires Vulkan >= "
				<< XR_VERSION_MAJOR(req.minApiVersionSupported) << "."
				<< XR_VERSION_MINOR(req.minApiVersionSupported) << "."
				<< XR_VERSION_PATCH(req.minApiVersionSupported) << std::endl;
		}

		// 1) Create VkInstance via XR
		VkInstance xrVkInstance = xrCreateVkInstance(g_openXRState.instance, g_openXRState.systemId);

		// 2) Ask XR for the physical device (enable2)
		PFN_xrGetVulkanGraphicsDevice2KHR pfnGetDevice2 = nullptr;
		xrGetInstanceProcAddr(g_openXRState.instance, "xrGetVulkanGraphicsDevice2KHR",
			reinterpret_cast<PFN_xrVoidFunction*>(&pfnGetDevice2));
		VkPhysicalDevice xrPhys = VK_NULL_HANDLE;
		if (pfnGetDevice2) {
			XrVulkanGraphicsDeviceGetInfoKHR info{ XR_TYPE_VULKAN_GRAPHICS_DEVICE_GET_INFO_KHR };
			info.systemId = g_openXRState.systemId;
			info.vulkanInstance = xrVkInstance;
			XR_CHECK(pfnGetDevice2(g_openXRState.instance, &info, &xrPhys));
		}
		else {
			// Legacy fallback
			PFN_xrGetVulkanGraphicsDeviceKHR pfnGetDevice = nullptr;
			xrGetInstanceProcAddr(g_openXRState.instance, "xrGetVulkanGraphicsDeviceKHR",
				reinterpret_cast<PFN_xrVoidFunction*>(&pfnGetDevice));
			if (!pfnGetDevice) {
				std::cerr << "Neither xrGetVulkanGraphicsDevice2KHR nor xrGetVulkanGraphicsDeviceKHR available" << std::endl;
				return false;
			}
			XR_CHECK(pfnGetDevice(g_openXRState.instance, g_openXRState.systemId, xrVkInstance, &xrPhys));
		}

		// 3) Find a graphics queue family
		uint32_t qCount = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(xrPhys, &qCount, nullptr);
		std::vector<VkQueueFamilyProperties> qprops(qCount);
		vkGetPhysicalDeviceQueueFamilyProperties(xrPhys, &qCount, qprops.data());
		uint32_t graphicsQ = UINT32_MAX;
		for (uint32_t i = 0; i < qCount; ++i) {
			if (qprops[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) { graphicsQ = i; break; }
		}
		if (graphicsQ == UINT32_MAX) {
			std::cerr << "No graphics queue family found" << std::endl;
			return false;
		}

		// 4) Create VkDevice via XR
		VkDevice xrVkDevice = xrCreateVkDevice(g_openXRState.instance, g_openXRState.systemId, xrPhys, graphicsQ);

		// 5) Populate queues
		VkQueue graphicsQueue = VK_NULL_HANDLE;
		vkGetDeviceQueue(xrVkDevice, graphicsQ, 0, &graphicsQueue);

		// 6) Replace your context handles with runtime-created ones
		context->m_instance = xrVkInstance;
		context->m_physicalDevice = xrPhys;
		context->m_device = xrVkDevice;
		context->m_queueGCT.familyIndex = graphicsQ;
		context->m_queueGCT.queueIndex = 0;
		context->m_queueGCT.queue = graphicsQueue;

		// 7) Create session with enable2 binding
		XrSessionCreateInfo sessionInfo{ XR_TYPE_SESSION_CREATE_INFO };
		XrGraphicsBindingVulkan2KHR vkBinding{ XR_TYPE_GRAPHICS_BINDING_VULKAN2_KHR };
		vkBinding.instance = context->m_instance;
		vkBinding.physicalDevice = context->m_physicalDevice;
		vkBinding.device = context->m_device;
		vkBinding.queueFamilyIndex = context->m_queueGCT.familyIndex;
		vkBinding.queueIndex = 0;
		sessionInfo.next = &vkBinding;
		sessionInfo.systemId = g_openXRState.systemId;

		std::cout << "Creating OpenXR session..." << std::endl;
		XR_CHECK(xrCreateSession(g_openXRState.instance, &sessionInfo, &g_openXRState.session));

		// 8) Reference space
		XrReferenceSpaceCreateInfo spaceInfo{ XR_TYPE_REFERENCE_SPACE_CREATE_INFO };
		spaceInfo.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_LOCAL;
		spaceInfo.poseInReferenceSpace = { {0,0,0,1}, {0,0,0} };
		XR_CHECK(xrCreateReferenceSpace(g_openXRState.session, &spaceInfo, &g_openXRState.referenceSpace));

		// 9) Begin session once (required before frame loop)
		XrSessionBeginInfo beginInfo{ XR_TYPE_SESSION_BEGIN_INFO };
		beginInfo.primaryViewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
		XR_CHECK(xrBeginSession(g_openXRState.session, &beginInfo));

		std::cout << "OpenXR initialization complete!" << std::endl;
		return true;
	}
	catch (const std::exception& e) {
		std::cerr << "OpenXR initialization exception: " << e.what() << std::endl;
		return false;
	}
}

VkDevice createSteamVRCompatibleDevice(VkPhysicalDevice physicalDevice, VkInstance instance)
{
	std::cout << "\n=== Creating SteamVR-Compatible Vulkan Device ===" << std::endl;

	// Get queue families
	uint32_t queueFamilyCount = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
	std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
	vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());

	// Find a graphics queue family
	uint32_t graphicsQueueFamilyIndex = UINT32_MAX;
	for (uint32_t i = 0; i < queueFamilyCount; ++i)
	{
		if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
		{
			graphicsQueueFamilyIndex = i;
			break;
		}
	}

	if (graphicsQueueFamilyIndex == UINT32_MAX)
	{
		throw std::runtime_error("No graphics queue family found");
	}

	// Create device with minimal extensions for SteamVR
	float queuePriority = 1.0f;
	VkDeviceQueueCreateInfo queueCreateInfo = {};
	queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
	queueCreateInfo.queueFamilyIndex = graphicsQueueFamilyIndex;
	queueCreateInfo.queueCount = 1;
	queueCreateInfo.pQueuePriorities = &queuePriority;

	// Minimal features for SteamVR
	VkPhysicalDeviceFeatures deviceFeatures = {};
	deviceFeatures.samplerAnisotropy = VK_TRUE;
	deviceFeatures.fillModeNonSolid = VK_TRUE;

	// SteamVR requires these extensions
	const char* deviceExtensions[] = {
		VK_KHR_SWAPCHAIN_EXTENSION_NAME,
		VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
		VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
		VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
		VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
		VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME,
	};

	VkDeviceCreateInfo deviceCreateInfo = {};
	deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	deviceCreateInfo.queueCreateInfoCount = 1;
	deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
	deviceCreateInfo.pEnabledFeatures = &deviceFeatures;
	deviceCreateInfo.enabledExtensionCount = sizeof(deviceExtensions) / sizeof(deviceExtensions[0]);
	deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions;

	VkDevice device;
	if (vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device) != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to create SteamVR-compatible Vulkan device");
	}

	std::cout << "Created SteamVR-compatible Vulkan device" << std::endl;
	return device;
}

std::shared_ptr<nvvk::Context> createSteamVRCompatibleContext()
{
	std::cout << "=== Creating SteamVR-Compatible Vulkan Context ===" << std::endl;

	nvvk::ContextCreateInfo vkSetup;
	vkSetup.apiMajor = 1;
	vkSetup.apiMinor = 2;  // Use 1.2 for better compatibility

	// SteamVR REQUIRES these extensions
	vkSetup.addDeviceExtension(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
	vkSetup.addDeviceExtension(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
	vkSetup.addDeviceExtension(VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);

	// For Windows
#ifdef _WIN32
	vkSetup.addDeviceExtension(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
	vkSetup.addDeviceExtension(VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
#endif

	vkSetup.addDeviceExtension(VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME);
	vkSetup.addDeviceExtension(VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME);

	// Instance extensions
	vkSetup.instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
	vkSetup.instanceExtensions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
	vkSetup.instanceExtensions.push_back(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);

	auto context = std::make_shared<nvvk::Context>();

	try {
		// Create instance
		context->initInstance(vkSetup);

		// Find a physical device
		uint32_t deviceCount = 0;
		vkEnumeratePhysicalDevices(context->m_instance, &deviceCount, nullptr);
		std::vector<VkPhysicalDevice> devices(deviceCount);
		vkEnumeratePhysicalDevices(context->m_instance, &deviceCount, devices.data());

		if (deviceCount == 0) {
			throw std::runtime_error("No Vulkan devices found");
		}

		// Select first discrete GPU or first available
		VkPhysicalDevice selectedDevice = VK_NULL_HANDLE;
		for (const auto& device : devices) {
			VkPhysicalDeviceProperties props;
			vkGetPhysicalDeviceProperties(device, &props);
			std::cout << "Found device: " << props.deviceName << std::endl;

			if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
				selectedDevice = device;
				std::cout << "  -> Selected discrete GPU: " << props.deviceName << std::endl;
				break;
			}
		}

		if (selectedDevice == VK_NULL_HANDLE) {
			selectedDevice = devices[0];
			VkPhysicalDeviceProperties props;
			vkGetPhysicalDeviceProperties(selectedDevice, &props);
			std::cout << "  -> Using first available: " << props.deviceName << std::endl;
		}

		context->m_physicalDevice = selectedDevice;

		// Create device with minimal features
		VkPhysicalDeviceFeatures features{};
		features.samplerAnisotropy = VK_TRUE;

		// Find graphics queue family
		uint32_t queueFamilyCount = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(context->m_physicalDevice, &queueFamilyCount, nullptr);
		std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
		vkGetPhysicalDeviceQueueFamilyProperties(context->m_physicalDevice, &queueFamilyCount, queueFamilies.data());

		uint32_t graphicsQueueFamily = UINT32_MAX;
		for (uint32_t i = 0; i < queueFamilyCount; ++i) {
			if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
				graphicsQueueFamily = i;
				std::cout << "  -> Graphics queue family: " << i << " (has "
					<< queueFamilies[i].queueCount << " queues)" << std::endl;
				break;
			}
		}

		if (graphicsQueueFamily == UINT32_MAX) {
			throw std::runtime_error("No graphics queue family found");
		}

		float queuePriority = 1.0f;
		VkDeviceQueueCreateInfo queueInfo{};
		queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		queueInfo.queueFamilyIndex = graphicsQueueFamily;
		queueInfo.queueCount = 1;
		queueInfo.pQueuePriorities = &queuePriority;

		std::vector<const char*> deviceExtensions = {
			VK_KHR_SWAPCHAIN_EXTENSION_NAME,
			VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
			VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
#ifdef _WIN32
			VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
			VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME,
#endif
			VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME,
			VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME,
		};

		VkDeviceCreateInfo deviceInfo{};
		deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		deviceInfo.queueCreateInfoCount = 1;
		deviceInfo.pQueueCreateInfos = &queueInfo;
		deviceInfo.pEnabledFeatures = &features;
		deviceInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
		deviceInfo.ppEnabledExtensionNames = deviceExtensions.data();

		std::cout << "Creating Vulkan device..." << std::endl;
		if (vkCreateDevice(context->m_physicalDevice, &deviceInfo, nullptr, &context->m_device) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create Vulkan device");
		}

		// Get queue
		vkGetDeviceQueue(context->m_device, graphicsQueueFamily, 0, &context->m_queueGCT.queue);
		context->m_queueGCT.familyIndex = graphicsQueueFamily;
		context->m_queueGCT.queueIndex = 0;

		std::cout << "SteamVR-compatible Vulkan context created successfully" << std::endl;

	}
	catch (const std::exception& e) {
		std::cerr << "Failed to create Vulkan context: " << e.what() << std::endl;
		// Don't throw here, let the caller handle it
		return nullptr;
	}

	return context;  // MAKE SURE THIS LINE IS PRESENT!
}

void getSystemOpenXR()
{
	XrSystemGetInfo systemGetInfo = { XR_TYPE_SYSTEM_GET_INFO };
	systemGetInfo.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;

	XrResult result = xrGetSystem(g_openXRState.instance, &systemGetInfo, &g_openXRState.systemId);
	if (result != XR_SUCCESS)
	{
		std::cerr << "xrGetSystem failed: " << result << std::endl;
		throw std::runtime_error("Failed to get OpenXR system");
	}

	xrGetSystemProperties(g_openXRState.instance, g_openXRState.systemId, &systemProperties);
	std::cout << "OpenXR system properties acquired successfully" << std::endl;
}

VkPhysicalDevice getOpenXRPhysicalDevice(XrInstance xrInstance, XrSystemId xrSystemId, VkInstance vkInstance) {
	std::cout << "=== getOpenXRPhysicalDevice ===" << std::endl;

	// Use Vulkan 1.0 API for SteamVR
	PFN_xrGetVulkanGraphicsRequirementsKHR pfnGetVulkanGraphicsRequirementsKHR = nullptr;
	XrResult resReq = xrGetInstanceProcAddr(xrInstance, "xrGetVulkanGraphicsRequirementsKHR",
		reinterpret_cast<PFN_xrVoidFunction*>(&pfnGetVulkanGraphicsRequirementsKHR));

	if (resReq != XR_SUCCESS || !pfnGetVulkanGraphicsRequirementsKHR) {
		std::cerr << "Vulkan 1.0 API not available!" << std::endl;
		return VK_NULL_HANDLE;
	}

	XrGraphicsRequirementsVulkanKHR vkReq{ XR_TYPE_GRAPHICS_REQUIREMENTS_VULKAN_KHR };
	XrResult r = pfnGetVulkanGraphicsRequirementsKHR(xrInstance, xrSystemId, &vkReq);
	if (r != XR_SUCCESS) {
		std::cerr << "xrGetVulkanGraphicsRequirementsKHR failed: " << r << std::endl;
		return VK_NULL_HANDLE;
	}

	std::cout << "Using Vulkan 1.0 API for SteamVR compatibility" << std::endl;

	// CORRECT: Use the actual field name from the structure definition
	std::cout << "  minApiVersionSupported: "
		<< XR_VERSION_MAJOR(vkReq.minApiVersionSupported) << "."
		<< XR_VERSION_MINOR(vkReq.minApiVersionSupported) << "."
		<< XR_VERSION_PATCH(vkReq.minApiVersionSupported) << std::endl;

	// List all available Vulkan devices
	uint32_t deviceCount = 0;
	vkEnumeratePhysicalDevices(vkInstance, &deviceCount, nullptr);
	std::vector<VkPhysicalDevice> devices(deviceCount);
	vkEnumeratePhysicalDevices(vkInstance, &deviceCount, devices.data());

	std::cout << "\nAvailable Vulkan physical devices (" << deviceCount << "):" << std::endl;

	VkPhysicalDevice selectedDevice = VK_NULL_HANDLE;
	const char* preferredDeviceNames[] = {
		"NVIDIA GeForce RTX 4060",
		"NVIDIA",
		"GeForce",
		"RTX",
	};

	for (uint32_t i = 0; i < deviceCount; ++i)
	{
		VkPhysicalDeviceProperties props;
		vkGetPhysicalDeviceProperties(devices[i], &props);
		std::cout << "  " << i << ": " << props.deviceName
			<< " (Handle: " << devices[i] << ")" << std::endl;

		// Prefer discrete GPUs
		if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
		{
			// Check against our preferred names
			for (const char* name : preferredDeviceNames)
			{
				if (strstr(props.deviceName, name) != nullptr)
				{
					selectedDevice = devices[i];
					std::cout << "  -> SELECTED: " << props.deviceName << " (matches: " << name << ")" << std::endl;
					break;
				}
			}
		}

		if (selectedDevice != VK_NULL_HANDLE)
		{
			break;
		}
	}

	// If no device found by name, use the first discrete GPU
	if (selectedDevice == VK_NULL_HANDLE)
	{
		for (uint32_t i = 0; i < deviceCount; ++i)
		{
			VkPhysicalDeviceProperties props;
			vkGetPhysicalDeviceProperties(devices[i], &props);

			if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
			{
				selectedDevice = devices[i];
				std::cout << "  -> FALLBACK SELECTED: " << props.deviceName << " (first discrete GPU)" << std::endl;
				break;
			}
		}
	}

	// fallback: 1st device
	if (selectedDevice == VK_NULL_HANDLE && deviceCount > 0)
	{
		selectedDevice = devices[0];
		VkPhysicalDeviceProperties props;
		vkGetPhysicalDeviceProperties(selectedDevice, &props);
		std::cout << "  -> ULTIMATE FALLBACK: " << props.deviceName << " (first available)" << std::endl;
	}

	if (selectedDevice == VK_NULL_HANDLE)
	{
		std::cerr << "ERROR: No suitable Vulkan device found!" << std::endl;
		return VK_NULL_HANDLE;
	}

	// Try to get device via OpenXR VULKAN2 API (but handle failures gracefully)
	PFN_xrGetVulkanGraphicsDevice2KHR pfnGetVulkanGraphicsDevice2KHR = nullptr;
	xrGetInstanceProcAddr(xrInstance, "xrGetVulkanGraphicsDevice2KHR",
		reinterpret_cast<PFN_xrVoidFunction*>(&pfnGetVulkanGraphicsDevice2KHR));

	if (pfnGetVulkanGraphicsDevice2KHR)
	{
		XrVulkanGraphicsDeviceGetInfoKHR deviceGetInfo{ XR_TYPE_VULKAN_GRAPHICS_DEVICE_GET_INFO_KHR };
		deviceGetInfo.systemId = xrSystemId;
		deviceGetInfo.vulkanInstance = vkInstance;

		VkPhysicalDevice xrDevice = VK_NULL_HANDLE;
		XrResult deviceResult = pfnGetVulkanGraphicsDevice2KHR(xrInstance, &deviceGetInfo, &xrDevice);
		std::cout << "xrGetVulkanGraphicsDevice2KHR result: " << deviceResult << std::endl;

		if (deviceResult == XR_SUCCESS && xrDevice != VK_NULL_HANDLE)
		{
			std::cout << "  OpenXR VULKAN2 provided device: " << xrDevice << std::endl;

			// Verify this device is in our list
			bool found = false;
			for (const auto& dev : devices)
			{
				if (dev == xrDevice)
				{
					found = true;
					break;
				}
			}

			if (found)
			{
				selectedDevice = xrDevice;
				std::cout << "  -> Using OpenXR's device selection" << std::endl;
			}
			else
			{
				std::cout << "  WARNING: OpenXR device not in Vulkan device list!" << std::endl;
			}
		}
		else
		{
			std::cout << "  NOTE: xrGetVulkanGraphicsDevice2KHR failed or returned null" << std::endl;
			std::cout << "  This is common with SteamVR on Linux - using manual selection" << std::endl;
		}
	}
	else
	{
		std::cout << "  NOTE: xrGetVulkanGraphicsDevice2KHR not available" << std::endl;
	}

	VkPhysicalDeviceProperties props;
	vkGetPhysicalDeviceProperties(selectedDevice, &props);
	std::cout << "\nFinal selected device: " << props.deviceName << " (Handle: " << selectedDevice << ")" << std::endl;

	return selectedDevice;
}



void cleanupOpenXR()
{
	if (g_openXRState.swapchain != XR_NULL_HANDLE)
	{
		xrDestroySwapchain(g_openXRState.swapchain);
		g_openXRState.swapchain = XR_NULL_HANDLE;
	}
	if (g_openXRState.session != XR_NULL_HANDLE)
	{
		xrDestroySession(g_openXRState.session);
		g_openXRState.session = XR_NULL_HANDLE;
	}
	if (g_openXRState.instance != XR_NULL_HANDLE)
	{
		xrDestroyInstance(g_openXRState.instance);
		g_openXRState.instance = XR_NULL_HANDLE;
	}
	std::cout << "OpenXR resources cleaned up" << std::endl;
}
bool isSteamVRReady()
{
	std::cout << "Checking if SteamVR is ready..." << std::endl;

	// Check if SteamVR process is running
	if (system("pgrep -x vrmonitor > /dev/null") != 0)
	{
		std::cerr << "SteamVR vrmonitor is not running!" << std::endl;
		return false;
	}

	// Check if VR server is running
	if (system("pgrep -x vrserver > /dev/null") != 0)
	{
		std::cerr << "SteamVR vrserver is not running!" << std::endl;
		return false;
	}

	// Check if ALVR is running (if using ALVR)
	if (system("pgrep -f alvr > /dev/null") != 0)
	{
		std::cerr << "ALVR is not running!" << std::endl;
		return false;
	}

	std::cout << "SteamVR/ALVR appears to be running" << std::endl;
	return true;
}

void verifyVrRuntime()
{
	std::cout << "\n=== Verifying VR Runtime ===" << std::endl;

#ifdef _WIN32
	// Windows-specific VR runtime checks
	std::cout << "Checking for SteamVR on Windows..." << std::endl;

	// Check common SteamVR installation paths
	const char* steamPaths[] = {
	  "C:\\Program Files (x86)\\Steam\\steamapps\\common\\SteamVR",
	  "C:\\Program Files\\Steam\\steamapps\\common\\SteamVR",
	  "F:\\Apps\\Steam\\steamapps\\common\\SteamVR"
	};

	bool steamVRFound = false;
	for (const auto& path : steamPaths) {
		DWORD attrs = GetFileAttributesA(path);
		if (attrs != INVALID_FILE_ATTRIBUTES && (attrs & FILE_ATTRIBUTE_DIRECTORY)) {
			steamVRFound = true;
			std::string runtimePath = std::string(path) + "\\steamxr_win64.json";

			if (GetFileAttributesA(runtimePath.c_str()) != INVALID_FILE_ATTRIBUTES) {
				std::cout << "Found SteamVR OpenXR runtime at: " << runtimePath << std::endl;
				_putenv_s("XR_RUNTIME_JSON", runtimePath.c_str());
				break;
			}
		}
	}

	if (!steamVRFound) {
		std::cout << "SteamVR not found in common locations." << std::endl;
		std::cout << "Please ensure SteamVR is installed from Steam." << std::endl;
	}

#else
	// Original Linux code
	const char* runtime = std::getenv("XR_RUNTIME_JSON");
	if (runtime)
	{
		std::cout << "Current XR_RUNTIME_JSON: " << runtime << std::endl;
		// ... rest of Linux code
	}
#endif

	std::cout << "VR Runtime check completed." << std::endl;
}


void createGraphicsBindingOpenXR(std::shared_ptr<nvvk::Context> m_context, VkPhysicalDevice xrPhysicalDevice)
{
	std::cout << "--Setting OPENXR graphics requirements---" << std::endl;

	// Create a separate SteamVR-compatible device
	VkDevice steamVRDevice = createSteamVRCompatibleDevice(xrPhysicalDevice, m_context->m_instance);

	// Use the SteamVR-compatible device for OpenXR
	memset(&g_openXRState.graphicsBinding, 0, sizeof(g_openXRState.graphicsBinding));
	g_openXRState.graphicsBinding.type = XR_TYPE_GRAPHICS_BINDING_VULKAN2_KHR;
	g_openXRState.graphicsBinding.next = nullptr;
	g_openXRState.graphicsBinding.instance = m_context->m_instance;
	g_openXRState.graphicsBinding.physicalDevice = xrPhysicalDevice;
	g_openXRState.graphicsBinding.device = steamVRDevice; // Use the compatible device
	g_openXRState.graphicsBinding.queueFamilyIndex = 0;   // Use first graphics queue family
	g_openXRState.graphicsBinding.queueIndex = 0;

	std::cout << "Using SteamVR-compatible Vulkan device for OpenXR" << std::endl;
}

void testMinimalOpenXRSession()
{
	std::cout << "\n=== Testing Minimal OpenXR Session ===" << std::endl;

	// Create the absolute minimal Vulkan instance
	VkApplicationInfo appInfo = {};
	appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	appInfo.pApplicationName = "MinimalTest";
	appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
	appInfo.pEngineName = "NoEngine";
	appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
	appInfo.apiVersion = VK_API_VERSION_1_0; // Use 1.0.0!

	const char* extensions[] = {
		VK_KHR_SURFACE_EXTENSION_NAME,
  #if defined(_WIN32)
		VK_KHR_WIN32_SURFACE_EXTENSION_NAME,
  #else
		VK_KHR_XCB_SURFACE_EXTENSION_NAME,
  #endif
	};

	VkInstanceCreateInfo instInfo = {};
	instInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	instInfo.pApplicationInfo = &appInfo;
	instInfo.enabledExtensionCount = 2;
	instInfo.ppEnabledExtensionNames = extensions;

	VkInstance vkInstance;
	if (vkCreateInstance(&instInfo, nullptr, &vkInstance) != VK_SUCCESS)
	{
		std::cout << "Failed to create minimal Vulkan instance" << std::endl;
		return;
	}

	// Get physical device
	uint32_t deviceCount = 0;
	vkEnumeratePhysicalDevices(vkInstance, &deviceCount, nullptr);
	std::vector<VkPhysicalDevice> devices(deviceCount);
	vkEnumeratePhysicalDevices(vkInstance, &deviceCount, devices.data());

	if (deviceCount == 0)
	{
		std::cout << "No Vulkan devices found" << std::endl;
		vkDestroyInstance(vkInstance, nullptr);
		return;
	}

	VkPhysicalDevice physicalDevice = devices[0];

	// Create minimal device
	float queuePriority = 1.0f;
	VkDeviceQueueCreateInfo queueCreateInfo = {};
	queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
	queueCreateInfo.queueFamilyIndex = 0;
	queueCreateInfo.queueCount = 1;
	queueCreateInfo.pQueuePriorities = &queuePriority;

	const char* deviceExts[] = {
		VK_KHR_SWAPCHAIN_EXTENSION_NAME,
	};

	VkDeviceCreateInfo deviceCreateInfo = {};
	deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	deviceCreateInfo.queueCreateInfoCount = 1;
	deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
	deviceCreateInfo.enabledExtensionCount = 1;
	deviceCreateInfo.ppEnabledExtensionNames = deviceExts;

	VkDevice device;
	if (vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device) != VK_SUCCESS)
	{
		std::cout << "Failed to create minimal Vulkan device" << std::endl;
		vkDestroyInstance(vkInstance, nullptr);
		return;
	}

	// Try to create OpenXR session with this minimal setup
	XrGraphicsBindingVulkanKHR vkBinding = { XR_TYPE_GRAPHICS_BINDING_VULKAN_KHR };
	vkBinding.instance = vkInstance;
	vkBinding.physicalDevice = physicalDevice;
	vkBinding.device = device;
	vkBinding.queueFamilyIndex = 0;
	vkBinding.queueIndex = 0;

	XrSessionCreateInfo sessionInfo = { XR_TYPE_SESSION_CREATE_INFO };
	sessionInfo.next = &vkBinding;
	sessionInfo.systemId = g_openXRState.systemId;

	XrSession testSession;
	XrResult result = xrCreateSession(g_openXRState.instance, &sessionInfo, &testSession);

	// After xrCreateSession call
	if (result != XR_SUCCESS) {
		std::cerr << "xrCreateSession failed with error: " << result << std::endl;

		// Check for common errors
		if (result == XR_ERROR_VALIDATION_FAILURE) {
			std::cerr << "  - XR_ERROR_VALIDATION_FAILURE: Check your Vulkan device/queue configuration" << std::endl;
		}
		else if (result == XR_ERROR_GRAPHICS_DEVICE_INVALID) {
			std::cerr << "  - XR_ERROR_GRAPHICS_DEVICE_INVALID: Vulkan device not compatible with OpenXR" << std::endl;
		}
		else if (result == XR_ERROR_RUNTIME_FAILURE) {
			std::cerr << "  - XR_ERROR_RUNTIME_FAILURE: SteamVR runtime issue" << std::endl;
		}
	}

	vkDestroyDevice(device, nullptr);
	vkDestroyInstance(vkInstance, nullptr);
}

void createSessionXR(std::shared_ptr<nvvk::Context> m_context)
{
	std::cout << "\n=== Creating OpenXR Session (Minimal Approach) ===" << std::endl;

	if (g_openXRState.instance == XR_NULL_HANDLE)
		throw std::runtime_error("OpenXR instance is null");
	if (g_openXRState.systemId == XR_NULL_SYSTEM_ID)
		throw std::runtime_error("OpenXR system ID is null");

	// Method 1: Try without graphics binding first (let OpenXR choose)
	std::cout << "\nMethod 1: Creating session without graphics binding..." << std::endl;
	{
		XrSessionCreateInfo sessionInfo = { XR_TYPE_SESSION_CREATE_INFO };
		sessionInfo.next = nullptr; // No graphics binding
		sessionInfo.systemId = g_openXRState.systemId;

		XrResult result = xrCreateSession(g_openXRState.instance, &sessionInfo, &g_openXRState.session);
		if (result == XR_SUCCESS)
		{
			std::cout << "SUCCESS! OpenXR created session without explicit graphics binding" << std::endl;
			std::cout << "This means OpenXR will handle graphics internally" << std::endl;
			return;
		}
		std::cout << "Method 1 failed: " << result << std::endl;
	}

	// Method 2: Try with minimal Vulkan 1.0 binding
	std::cout << "\nMethod 2: Creating session with minimal Vulkan 1.0 binding..." << std::endl;
	{
		XrGraphicsBindingVulkanKHR vkBinding = { XR_TYPE_GRAPHICS_BINDING_VULKAN_KHR };
		vkBinding.instance = m_context->m_instance;
		vkBinding.physicalDevice = m_context->m_physicalDevice;
		vkBinding.device = m_context->m_device;
		vkBinding.queueFamilyIndex = 0;
		vkBinding.queueIndex = 0;

		XrSessionCreateInfo sessionInfo = { XR_TYPE_SESSION_CREATE_INFO };
		sessionInfo.next = &vkBinding;
		sessionInfo.systemId = g_openXRState.systemId;

		XrResult result = xrCreateSession(g_openXRState.instance, &sessionInfo, &g_openXRState.session);
		if (result == XR_SUCCESS)
		{
			std::cout << "SUCCESS with Vulkan 1.0 binding!" << std::endl;
			// Convert to Vulkan2 for consistency
			g_openXRState.graphicsBinding.type = XR_TYPE_GRAPHICS_BINDING_VULKAN2_KHR;
			g_openXRState.graphicsBinding.next = nullptr;
			g_openXRState.graphicsBinding.instance = m_context->m_instance;
			g_openXRState.graphicsBinding.physicalDevice = m_context->m_physicalDevice;
			g_openXRState.graphicsBinding.device = m_context->m_device;
			g_openXRState.graphicsBinding.queueFamilyIndex = 0;
			g_openXRState.graphicsBinding.queueIndex = 0;
			return;
		}
		std::cout << "Method 2 failed: " << result << std::endl;
	}

	// Method 3: Try with the exact same approach as hello_xr
	std::cout << "\nMethod 3: Creating session using hello_xr approach..." << std::endl;
	{
		// This is the approach used by the official hello_xr sample
		XrGraphicsBindingVulkanKHR vkBinding = { XR_TYPE_GRAPHICS_BINDING_VULKAN_KHR };
		vkBinding.instance = m_context->m_instance;
		vkBinding.physicalDevice = m_context->m_physicalDevice;
		vkBinding.device = m_context->m_device;

		// IMPORTANT: Get the queue from the device to ensure it's valid
		VkQueue queue;
		vkGetDeviceQueue(m_context->m_device, 0, 0, &queue);
		(void)queue; // Use it to avoid unused variable warning

		vkBinding.queueFamilyIndex = 0;
		vkBinding.queueIndex = 0;

		XrSessionCreateInfo sessionInfo = { XR_TYPE_SESSION_CREATE_INFO };
		sessionInfo.next = &vkBinding;
		sessionInfo.systemId = g_openXRState.systemId;

		XrResult result = xrCreateSession(g_openXRState.instance, &sessionInfo, &g_openXRState.session);
		if (result == XR_SUCCESS)
		{
			std::cout << "SUCCESS with hello_xr approach!" << std::endl;
			g_openXRState.graphicsBinding.type = XR_TYPE_GRAPHICS_BINDING_VULKAN2_KHR;
			g_openXRState.graphicsBinding.next = nullptr;
			g_openXRState.graphicsBinding.instance = m_context->m_instance;
			g_openXRState.graphicsBinding.physicalDevice = m_context->m_physicalDevice;
			g_openXRState.graphicsBinding.device = m_context->m_device;
			g_openXRState.graphicsBinding.queueFamilyIndex = 0;
			g_openXRState.graphicsBinding.queueIndex = 0;
			return;
		}
		std::cout << "Method 3 failed: " << result << std::endl;
	}

	// Method 4: Check if it's an ALVR-specific issue
	std::cout << "\nMethod 4: Checking ALVR compatibility..." << std::endl;
	{
		// ALVR sometimes needs special handling
		// Check if ALVR is running
		if (system("pgrep -f alvr_server > /dev/null") == 0)
		{
			std::cout << "ALVR server detected, trying compatibility mode..." << std::endl;

			// Try creating a session with a delay between attempts
			for (int i = 0; i < 5; i++)
			{
				XrGraphicsBindingVulkanKHR vkBinding = { XR_TYPE_GRAPHICS_BINDING_VULKAN_KHR };
				vkBinding.instance = m_context->m_instance;
				vkBinding.physicalDevice = m_context->m_physicalDevice;
				vkBinding.device = m_context->m_device;
				vkBinding.queueFamilyIndex = 0;
				vkBinding.queueIndex = 0;

				XrSessionCreateInfo sessionInfo = { XR_TYPE_SESSION_CREATE_INFO };
				sessionInfo.next = &vkBinding;
				sessionInfo.systemId = g_openXRState.systemId;

				XrResult result = xrCreateSession(g_openXRState.instance, &sessionInfo, &g_openXRState.session);
				if (result == XR_SUCCESS)
				{
					std::cout << "SUCCESS on attempt " << (i + 1) << "!" << std::endl;
					g_openXRState.graphicsBinding.type = XR_TYPE_GRAPHICS_BINDING_VULKAN2_KHR;
					g_openXRState.graphicsBinding.next = nullptr;
					g_openXRState.graphicsBinding.instance = m_context->m_instance;
					g_openXRState.graphicsBinding.physicalDevice = m_context->m_physicalDevice;
					g_openXRState.graphicsBinding.device = m_context->m_device;
					g_openXRState.graphicsBinding.queueFamilyIndex = 0;
					g_openXRState.graphicsBinding.queueIndex = 0;
					return;
				}

				std::cout << "Attempt " << (i + 1) << " failed: " << result << std::endl;
				if (i < 4)
				{
					usleep(500000); // 0.5 second delay
				}
			}
		}
	}

	throw std::runtime_error("All session creation methods failed");
}

// void createSessionXR(std::shared_ptr<nvvk::Context> m_context)
// {
//   std::cout << "\n=== Creating OpenXR Session ===" << std::endl;

//   // First, verify OpenXR instance and system
//   if (xrInstance == XR_NULL_HANDLE)
//   {
//     throw std::runtime_error("OpenXR instance is null");
//   }

//   if (xrSystemId == XR_NULL_SYSTEM_ID)
//   {
//     throw std::runtime_error("OpenXR system ID is null");
//   }

//   std::cout << "Graphics Binding Structure:" << std::endl;
//   std::cout << "  Type: " << graphicsBinding.type << std::endl;
//   std::cout << "  Next: " << graphicsBinding.next << std::endl;
//   std::cout << "  Instance: " << (void *)graphicsBinding.instance << std::endl;
//   std::cout << "  PhysicalDevice: " << (void *)graphicsBinding.physicalDevice << std::endl;
//   std::cout << "  Device: " << (void *)graphicsBinding.device << std::endl;

//   // Try up to 3 times with delays
//   for (int attempt = 1; attempt <= 3; attempt++)
//   {
//     std::cout << "\nAttempt " << attempt << " to create session..." << std::endl;

//     // Re-initialize graphics binding each time
//     XrGraphicsBindingVulkan2KHR localBinding = {
//         .type = XR_TYPE_GRAPHICS_BINDING_VULKAN2_KHR,
//         .next = nullptr,
//         .instance = m_context->m_instance,
//         .physicalDevice = m_context->m_physicalDevice,
//         .device = m_context->m_device,
//         .queueFamilyIndex = m_context->m_queueGCT.familyIndex,
//         .queueIndex = 0};

//     XrSessionCreateInfo sessionCreateInfo = {XR_TYPE_SESSION_CREATE_INFO};
//     sessionCreateInfo.next = &localBinding;
//     sessionCreateInfo.systemId = xrSystemId;

//     XrResult result = xrCreateSession(xrInstance, &sessionCreateInfo, &xrSession);

//     if (result == XR_SUCCESS)
//     {
//       std::cout << "OpenXR session created successfully on attempt " << attempt << std::endl;
//       // Update the global binding
//       graphicsBinding = localBinding;
//       return;
//     }

//     std::cerr << "Attempt " << attempt << " failed: " << result << std::endl;

//     if (attempt < 3)
//     {
//       std::cout << "Waiting 1 second before retry..." << std::endl;
//       // Simple sleep (use usleep for microseconds)
//       usleep(1000000); // 1 second
//     }
//   }

//   // If we get here, all attempts failed
//   std::cerr << "\n=== All session creation attempts failed ===" << std::endl;
//   throw std::runtime_error("Failed to create OpenXR session after 3 attempts");
// }

namespace nvvkhl
{
	//////////////////////////////////////////////////////////////////////////
	/// </summary> Ray trace multiple primitives
	class OptixDenoiserEngine : public nvvkhl::IAppElement
	{
		enum GbufferNames
		{
			eGBufLdr,
			eGBufResult,
			eGBufAlbedo,
			eGBufNormal,
			eGBufDepth,
			eGBufDisparity,
			eGbufDenoised,
		};

		struct Settings
		{
			int maxFrames{ 200000 };
			int maxSamples{ 4 };
			int maxDepth{ 4 };
			bool showAxis{ true };
			glm::vec4 clearColor{ 1.F };
			float envRotation{ -130.F };
			bool denoiseApply{ true };
			bool denoiseFirstFrame{ true };
			int denoiseEveryNFrames{ 1 };
		} m_settings;

	public:
		OptixDenoiserEngine()
		{
			//m_frameInfo.maxLuminance = 10.0F;
			m_frameInfo.maxLuminance = 500.0F;
			m_frameInfo.clearColor = glm::vec4(1.F);
		};

		~OptixDenoiserEngine() override = default;

		void onAttach(nvvkhl::Application* app) override
		{
			m_app = app;
			m_device = m_app->getDevice();
			m_physicalDevice = m_app->getPhysicalDevice();

			VmaAllocatorCreateInfo allocator_info = {};
			allocator_info.physicalDevice = app->getPhysicalDevice();
			allocator_info.device = app->getDevice();
			allocator_info.instance = app->getInstance();
			allocator_info.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;

			m_dutil = std::make_unique<nvvk::DebugUtil>(m_device);                              // Debug utility
			m_alloc = std::make_unique<AllocVma>(allocator_info);                               // Allocator
			m_scene = std::make_unique<nvh::gltf::Scene>();                                     // GLTF scene
			m_sceneVk = std::make_unique<SceneVk>(m_device, m_physicalDevice, m_alloc.get());   // GLTF Scene buffers
			m_sceneRtx = std::make_unique<SceneRtx>(m_device, m_physicalDevice, m_alloc.get()); // GLTF Scene BLAS/TLAS
			m_tonemapper = std::make_unique<TonemapperPostProcess>(m_device, m_alloc.get());
			m_sbt = std::make_unique<nvvk::SBTWrapper>();
			m_picker = std::make_unique<nvvk::RayPickerKHR>(m_device, m_physicalDevice, m_alloc.get());
			m_hdrEnv = std::make_unique<HdrEnv>(m_device, m_physicalDevice, m_alloc.get());
			m_rtxSet = std::make_unique<nvvk::DescriptorSetContainer>(m_device);
			m_sceneSet = std::make_unique<nvvk::DescriptorSetContainer>(m_device);

			// Override the way benchmark count frames, to only use valid ones
			g_elemBenchmark->setCurrentFrame([&]
				{ return m_frame; });

#ifdef NVP_SUPPORTS_OPTIX7
			m_denoiser = std::make_unique<DenoiserOptix>();
			m_denoiser->setup(m_device, m_physicalDevice, m_app->getQueue(0).familyIndex);

			OptixDenoiserOptions d_options;
			d_options.guideAlbedo = 1u;
			d_options.guideNormal = 1u;
			m_denoiser->initOptiX(d_options, OPTIX_PIXEL_FORMAT_FLOAT4, true);
			m_denoiser->createSemaphore();
			m_denoiser->createCopyPipeline();
#else
			m_settings.denoiseApply = false;
			LOGE("OptiX is not supported");
#endif // NVP_SUPPORTS_OPTIX7

			m_hdrEnv->loadEnvironment("");

			// Requesting ray tracing properties
			VkPhysicalDeviceRayTracingPipelinePropertiesKHR rt_prop{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR };
			VkPhysicalDeviceProperties2 prop2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
			prop2.pNext = &rt_prop;
			vkGetPhysicalDeviceProperties2(m_app->getPhysicalDevice(), &prop2);
			// Create utilities to create the Shading Binding Table (SBT)
			uint32_t gct_queue_index = m_app->getQueue(0).familyIndex;
			m_sbt->setup(m_app->getDevice(), gct_queue_index, m_alloc.get(), rt_prop);

			// Set initial view size to XR resolution to avoid redundant resizes
			if (m_enableXR && g_openXRState.swapchainWidth > 0)
			{
				m_viewSize = glm::vec2(g_openXRState.swapchainWidth, g_openXRState.swapchainHeight);
			}
			// Create resources
			createCommandBuffers();
			createGbuffers(m_viewSize);
			createVulkanBuffers();

			m_tonemapper->createComputePipeline();
		}

		void onDetach() override
		{
			vkDeviceWaitIdle(m_device);
			destroyResources();
		}

		void onResize(uint32_t width, uint32_t height) override
		{
			if (m_enableXR && g_openXRState.swapchainWidth > 0) {
				width = g_openXRState.swapchainWidth;
				height = g_openXRState.swapchainHeight;
			}
			// Skip if size hasn't actually changed
			if (m_gBuffers && m_gBuffers->getSize().width == width && m_gBuffers->getSize().height == height)
			{
				return;
			}
			std::cout << "Resizing for " << (m_enableXR ? "XR" : "desktop") << ": " << width << "x" << height << std::endl;
			createGbuffers({ width, height });

			m_tonemapper->updateComputeDescriptorSets(
				m_gBuffers->getDescriptorImageInfo(showDenoisedImage() ? eGbufDenoised : eGBufResult),
				m_gBuffers->getDescriptorImageInfo(eGBufLdr));

			writeRtxSet();
		}



		void onUIMenu() override
		{
			bool load_file{ false };

			windowTitle();

			if (ImGui::BeginMenu("File"))
			{
				if (ImGui::MenuItem("Load", "Ctrl+O"))
				{
					load_file = true;
				}
				ImGui::Separator();
				ImGui::EndMenu();
			}
			if (ImGui::IsKeyPressed(ImGuiKey_O) && ImGui::IsKeyDown(ImGuiKey_LeftCtrl))
			{
				load_file = true;
			}

			if (load_file)
			{
				auto filename = NVPSystem::windowOpenFileDialog(m_app->getWindowHandle(), "Load glTF | HDR",
					"glTF(.gltf, .glb), HDR(.hdr)|*.gltf;*.glb;*.hdr");
				onFileDrop(filename.c_str());
			}
		}

		void onFileDrop(const char* filename) override
		{
			namespace fs = std::filesystem;
			vkDeviceWaitIdle(m_device);
			std::string extension = fs::path(filename).extension().string();
			if (extension == ".gltf" || extension == ".glb")
			{
				createScene(filename);
			}
			else if (extension == ".hdr")
			{
				createHdr(filename);
				resetFrame();
			}

			resetFrame();
		}

		void onUIRender() override
		{
			using namespace ImGuiH;

			bool reset{ false };
			// Pick under mouse cursor
			if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left) || ImGui::IsKeyPressed(ImGuiKey_Space))
			{
				screenPicking();
			}
			if (ImGui::IsKeyPressed(ImGuiKey_M))
			{
				onResize(m_app->getViewportSize().width, m_app->getViewportSize().height); // Force recreation of G-Buffers
				reset = true;
			}

			{ // Setting menu
				ImGui::Begin("Settings");

				if (ImGui::CollapsingHeader("Camera"))
				{
					ImGuiH::CameraWidget();
				}

				if (ImGui::CollapsingHeader("Settings"))
				{
					PropertyEditor::begin();

					if (PropertyEditor::treeNode("Ray Tracing"))
					{
						reset |= PropertyEditor::entry("Depth", [&]
							{ return ImGui::SliderInt("#1", &m_settings.maxDepth, 1, 10); });
						reset |= PropertyEditor::entry("Samples", [&]
							{ return ImGui::SliderInt("#2", &m_settings.maxSamples, 1, 5); });
						reset |= PropertyEditor::entry("Frames",
							[&]
							{ return ImGui::DragInt("#3", &m_settings.maxFrames, 5.0F, 1, 1000000); });
						PropertyEditor::treePop();
					}
					PropertyEditor::entry("Show Axis", [&]
						{ return ImGui::Checkbox("##4", &m_settings.showAxis); });
					PropertyEditor::end();
				}

				if (ImGui::CollapsingHeader("Environment"))
				{
					PropertyEditor::begin();
					if (PropertyEditor::treeNode("Hdr"))
					{
						reset |= PropertyEditor::entry(
							"Color", [&]
							{ return ImGui::ColorEdit3("##Color", &m_settings.clearColor.x, ImGuiColorEditFlags_Float); },
							"Color multiplier");

						reset |= PropertyEditor::entry(
							"Rotation", [&]
							{ return ImGui::SliderAngle("Rotation", &m_settings.envRotation); }, "Rotating the environment");
						PropertyEditor::treePop();
					}
					PropertyEditor::end();
				}

				if (ImGui::CollapsingHeader("Tonemapper"))
				{
					m_tonemapper->onUI();
				}

				// #OPTIX_D
				if (ImGui::CollapsingHeader("Denoiser", ImGuiTreeNodeFlags_DefaultOpen))
				{
					ImGui::Checkbox("Denoise", &m_settings.denoiseApply);
					ImGui::Checkbox("First Frame", &m_settings.denoiseFirstFrame);
					ImGui::SliderInt("N-frames", &m_settings.denoiseEveryNFrames, 1, 500);
					ImGui::SliderFloat("Blend", &m_blendFactor, 0.f, 1.0f);
					ImGui::SliderFloat("Middle Radius", &m_middleRadius, 0.1f, 1.0f);
					int denoised_frame = -1;
					if (m_settings.denoiseApply)
					{
						if (m_frame >= m_settings.maxFrames)
							denoised_frame = m_settings.maxFrames;
						else if (m_settings.denoiseFirstFrame && (m_frame < m_settings.denoiseEveryNFrames))
							denoised_frame = 0;
						else if (m_frame >= m_settings.denoiseEveryNFrames)
							denoised_frame = (m_frame / m_settings.denoiseEveryNFrames) * m_settings.denoiseEveryNFrames;
					}
					ImGui::Text("Denoised Frame: %d", denoised_frame);

					ImVec2 tumbnailSize = { 150 * m_gBuffers->getAspectRatio(), 150 };
					ImGui::Text("Albedo");
					ImGui::Image(m_gBuffers->getDescriptorSet(eGBufAlbedo), tumbnailSize);
					ImGui::Text("Normal");
					ImGui::Image(m_gBuffers->getDescriptorSet(eGBufNormal), tumbnailSize);
					ImGui::Text("Depth");
					ImGui::Image(m_gBuffers->getDescriptorSet(eGBufDepth), tumbnailSize);
					ImGui::Text("Disparity");
					ImGui::Image(m_gBuffers->getDescriptorSet(eGBufDisparity), tumbnailSize);
					ImGui::Text("Result");
					ImGui::Image(m_gBuffers->getDescriptorSet(eGBufResult), tumbnailSize);
					ImGui::Text("Denoised");
					ImGui::Image(m_gBuffers->getDescriptorSet(eGbufDenoised), tumbnailSize);
				}

				ImGui::End();

				if (reset)
				{
					resetFrame();
				}
			}

			m_tonemapper->updateComputeDescriptorSets(m_gBuffers->getDescriptorImageInfo(showDenoisedImage() ? eGbufDenoised : eGBufResult),
				m_gBuffers->getDescriptorImageInfo(eGBufLdr));

			{ // Rendering Viewport
				ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0F, 0.0F));
				ImGui::Begin("Viewport");

				// Display the G-Buffer image
				ImGui::Image(m_gBuffers->getDescriptorSet(eGBufLdr), ImGui::GetContentRegionAvail());

				if (m_settings.showAxis)
				{ // Display orientation axis at the bottom left corner of the window
					const float axisSize = 25.F;
					ImVec2 pos = ImGui::GetWindowPos();
					pos.y += ImGui::GetWindowSize().y;
					pos += ImVec2(axisSize * 1.1F, -axisSize * 1.1F) * ImGui::GetWindowDpiScale(); // Offset
					ImGuiH::Axis(pos, CameraManip.getMatrix(), axisSize);
				}

				ImGui::End();
				ImGui::PopStyleVar();
			}
		}

		glm::mat4 xrPoseToMat4(const XrPosef& pose)
		{
			glm::quat orientation(pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z);
			glm::mat4 rotation = glm::mat4_cast(orientation);
			glm::mat4 translation = glm::translate(glm::mat4(1.0f), glm::vec3(pose.position.x, pose.position.y, pose.position.z));
			return translation * rotation;
		}

		glm::mat4 xrFovToProjMatrix(const XrFovf& fov, float nearZ, float farZ)
		{
			float tanLeft = tanf(fov.angleLeft);
			float tanRight = tanf(fov.angleRight);
			float tanUp = tanf(fov.angleUp);
			float tanDown = tanf(fov.angleDown);

			glm::mat4 proj = glm::mat4(0.0f);
			proj[0][0] = 2.0f / (tanRight - tanLeft);
			proj[1][1] = 2.0f / (tanUp - tanDown);
			proj[2][2] = -(farZ + nearZ) / (farZ - nearZ);
			proj[2][3] = -1.0f;
			proj[3][2] = -(2.0f * farZ * nearZ) / (farZ - nearZ);
			proj[0][2] = (tanRight + tanLeft) / (tanRight - tanLeft);
			proj[1][2] = (tanUp + tanDown) / (tanUp - tanDown);
			return proj;
		}

		// XR Raytracing render
		void onRenderVR(VkCommandBuffer cmd)
		{
			if (!m_scene->valid())
				return;

			// Using local command buffer for the frame, same as non-VR path
			const CommandFrame& commandFrame = m_commandFrames[m_app->getFrameCycleIndex()];
			VkCommandBuffer vkCmd = commandFrame.cmdBuffer[0];

			VkResult resetResult = vkResetCommandPool(m_device, commandFrame.cmdPool, 0);
			if (resetResult != VK_SUCCESS)
				return;

			VkCommandBufferBeginInfo vkBeginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
			vkBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
			if (vkBeginCommandBuffer(vkCmd, &vkBeginInfo) != VK_SUCCESS)
				return;

			// Ensure GBuffers match XR swapchain size
			// uint32_t requiredWidth = g_openXRState.swapchainWidth * 2;
			// uint32_t requiredHeight = g_openXRState.swapchainHeight;
			if (m_gBuffers->getSize().width != g_openXRState.swapchainWidth ||
				m_gBuffers->getSize().height != g_openXRState.swapchainHeight) {
				vkEndCommandBuffer(vkCmd);
				onResize(g_openXRState.swapchainWidth, g_openXRState.swapchainHeight);
				// Submit an empty frame to OpenXR
				XrFrameWaitInfo waitInfoResize{ XR_TYPE_FRAME_WAIT_INFO };
				XrFrameState stateResize{ XR_TYPE_FRAME_STATE };
				xrWaitFrame(g_openXRState.session, &waitInfoResize, &stateResize);
				XrFrameBeginInfo beginResize{ XR_TYPE_FRAME_BEGIN_INFO };
				xrBeginFrame(g_openXRState.session, &beginResize);
				XrFrameEndInfo endInfo{ XR_TYPE_FRAME_END_INFO };
				endInfo.displayTime = stateResize.predictedDisplayTime;
				endInfo.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
				endInfo.layerCount = 0;
				endInfo.layers = nullptr;
				XR_CHECK(xrEndFrame(g_openXRState.session, &endInfo));
				return;
			}

			// 1) Wait + Begin frame
			XrFrameWaitInfo waitInfo{ XR_TYPE_FRAME_WAIT_INFO };
			XrFrameState xrFrameState{ XR_TYPE_FRAME_STATE };
			XR_CHECK(xrWaitFrame(g_openXRState.session, &waitInfo, &xrFrameState));

			XrFrameBeginInfo xrBeginInfo{ XR_TYPE_FRAME_BEGIN_INFO };
			XR_CHECK(xrBeginFrame(g_openXRState.session, &xrBeginInfo));

			if (!xrFrameState.shouldRender) {
				vkEndCommandBuffer(vkCmd);
				XrFrameEndInfo endInfo{ XR_TYPE_FRAME_END_INFO };
				endInfo.next = nullptr;
				endInfo.displayTime = xrFrameState.predictedDisplayTime;
				endInfo.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
				endInfo.layerCount = 0;
				endInfo.layers = nullptr;
				XR_CHECK(xrEndFrame(g_openXRState.session, &endInfo));
				return;
			}

			// 2) Locate views
			XrView views[2] = { {XR_TYPE_VIEW}, {XR_TYPE_VIEW} };
			uint32_t viewCountOutput = 0;

			XrViewLocateInfo viewLocateInfo{ XR_TYPE_VIEW_LOCATE_INFO };
			viewLocateInfo.viewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
			viewLocateInfo.displayTime = xrFrameState.predictedDisplayTime;
			viewLocateInfo.space = g_openXRState.referenceSpace;

			XrViewState xrViewState{ XR_TYPE_VIEW_STATE };
			XR_CHECK(xrLocateViews(g_openXRState.session, &viewLocateInfo, &xrViewState, 2, &viewCountOutput, views));

			const XrViewStateFlags validMask = XR_VIEW_STATE_ORIENTATION_VALID_BIT | XR_VIEW_STATE_POSITION_VALID_BIT;
			if (viewCountOutput != 2 || (xrViewState.viewStateFlags & validMask) != validMask)
			{
				vkEndCommandBuffer(vkCmd);
				XrFrameEndInfo endInfo{ XR_TYPE_FRAME_END_INFO };
				endInfo.next = nullptr;
				endInfo.displayTime = xrFrameState.predictedDisplayTime;
				endInfo.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
				endInfo.layerCount = 0;
				endInfo.layers = nullptr;
				XR_CHECK(xrEndFrame(g_openXRState.session, &endInfo));
				return;
			}

			// Fill per-eye FrameInfo: use CameraManip as the scene base,
			// apply OpenXR head tracking as relative offset
			{
				const float nearZ = 0.1f;
				const float farZ = 1000.0f;

				// Get the desktop camera transform as our scene base position
				glm::vec3 sceneEye = CameraManip.getEye();
				glm::vec3 sceneCenter = CameraManip.getCenter();
				glm::vec3 sceneUp = CameraManip.getUp();

				// Build a scene-space transform: position at sceneEye, looking toward sceneCenter
				glm::mat4 sceneBaseMat = glm::inverse(glm::lookAt(sceneEye, sceneCenter, sceneUp));

				// Left eye: combine scene base with XR head pose
				glm::mat4 xrLeftPose = xrPoseToMat4(views[0].pose);
				glm::mat4 leftEyeWorld = sceneBaseMat * xrLeftPose;
				glm::mat4 leftViewMat = glm::inverse(leftEyeWorld);
				glm::mat4 leftProjMat = xrFovToProjMatrix(views[0].fov, nearZ, farZ);
				leftProjMat[1][1] *= -1;

				m_frameInfo.view = leftViewMat;
				m_frameInfo.proj = leftProjMat;
				m_frameInfo.envRotation = m_settings.envRotation;
				m_frameInfo.clearColor = m_settings.clearColor;
				m_frameInfo.camPos = glm::vec4(glm::vec3(leftEyeWorld[3]), 0.0f);

				// Right eye: combine scene base with XR head pose
				glm::mat4 xrRightPose = xrPoseToMat4(views[1].pose);
				glm::mat4 rightEyeWorld = sceneBaseMat * xrRightPose;
				glm::mat4 rightViewMat = glm::inverse(rightEyeWorld);
				glm::mat4 rightProjMat = xrFovToProjMatrix(views[1].fov, nearZ, farZ);
				rightProjMat[1][1] *= -1;

				m_frameInfo.view2 = rightViewMat;
				m_frameInfo.proj2 = rightProjMat;
				m_frameInfo.camPos2 = glm::vec4(glm::vec3(rightEyeWorld[3]), 0.0f);
				
				// Compute real IPD from XR eye poses (distance between left and right eye positions)
				glm::vec3 leftEyePos(views[0].pose.position.x, views[0].pose.position.y, views[0].pose.position.z);
				glm::vec3 rightEyePos(views[1].pose.position.x, views[1].pose.position.y, views[1].pose.position.z);
				m_xrEyeSeparation = glm::distance(leftEyePos, rightEyePos);
			}

			// VR always resets frame to 0 — each frame is a new view, no accumulation
			m_frame = 0;

			// 3) Acquire swapchain image
			uint32_t swapchainImageIndex = 0;
			XrSwapchainImageAcquireInfo acquireInfo{ XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO };
			XR_CHECK(xrAcquireSwapchainImage(g_openXRState.swapchain, &acquireInfo, &swapchainImageIndex));

			XrSwapchainImageWaitInfo swWaitInfo{ XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO };
			swWaitInfo.next = nullptr;
			swWaitInfo.timeout = XR_INFINITE_DURATION;
			XR_CHECK(xrWaitSwapchainImage(g_openXRState.swapchain, &swWaitInfo));

			// 4) Transition swapchain to TRANSFER_DST_OPTIMAL
			{
				const uint32_t idx = swapchainImageIndex;
				VkImage swapImg = g_openXRState.swapchainImages[idx].image;
				VkImageLayout oldLayout = g_openXRState.swapchainImageLayouts[idx];

				VkImageMemoryBarrier toDst{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
				toDst.oldLayout = oldLayout;
				toDst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
				toDst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
				toDst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
				toDst.image = swapImg;
				toDst.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 2 };
				toDst.srcAccessMask = 0;
				toDst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

				vkCmdPipelineBarrier(vkCmd,
					VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
					VK_PIPELINE_STAGE_TRANSFER_BIT,
					0, 0, nullptr, 0, nullptr, 1, &toDst);

				g_openXRState.swapchainImageLayouts[idx] = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			}

			// 5) Render offscreen — use full GPU power
			vkCmdUpdateBuffer(vkCmd, m_bFrameInfo.buffer, 0, sizeof(FrameInfo), &m_frameInfo);

			m_pushConst.maxDepth = m_settings.maxDepth;
			m_pushConst.maxSamples = m_settings.maxSamples;
			m_pushConst.frame = m_frame;
			m_pushConst.middleRadius = m_middleRadius;
			m_pushConst.eyeSeparation = m_xrEyeSeparation;

			raytraceScene(vkCmd);


#ifdef NVP_SUPPORTS_OPTIX7
			// Denoise in VR — must sync Vulkan→CUDA→Vulkan inline
			if (m_settings.denoiseApply)
			{
				copyImagesToCuda(vkCmd);
				vkEndCommandBuffer(vkCmd);

				// Signal CUDA that Vulkan is done
				VkSemaphoreSubmitInfoKHR signal_sem{
					.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO_KHR,
					.semaphore = m_denoiser->getTLSemaphore(),
					.value = ++m_fenceValue,
					.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT_KHR,
				};
				VkCommandBufferSubmitInfoKHR cmd_info{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO_KHR, nullptr, vkCmd };
				VkSubmitInfo2KHR submit{
					.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2_KHR,
					.commandBufferInfoCount = 1,
					.pCommandBufferInfos = &cmd_info,
					.signalSemaphoreInfoCount = 1,
					.pSignalSemaphoreInfos = &signal_sem,
				};
				vkQueueSubmit2(m_app->getQueue(0).queue, 1, &submit, VK_NULL_HANDLE);

				// Run OptiX denoiser on CUDA side
				denoiseImage();

				// Wait for CUDA to finish
				VkSemaphore tlSemaphore = m_denoiser->getTLSemaphore();
				VkSemaphoreWaitInfo waitInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO };
				waitInfo.semaphoreCount = 1;
				waitInfo.pSemaphores = &tlSemaphore;
				waitInfo.pValues = &m_fenceValue;
				vkWaitSemaphores(m_device, &waitInfo, UINT64_MAX);

				// Start new command buffer for the rest of the frame
				vkCmd = commandFrame.cmdBuffer[1];
				VkCommandBufferBeginInfo beginInfo2{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
				beginInfo2.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
				vkBeginCommandBuffer(vkCmd, &beginInfo2);

				copyCudaImagesToVulkan(vkCmd);
			}
#endif
			// Use denoised image for tonemapping in VR
			m_tonemapper->updateComputeDescriptorSets(
				m_gBuffers->getDescriptorImageInfo(
					m_settings.denoiseApply ? eGbufDenoised : eGBufResult),
				m_gBuffers->getDescriptorImageInfo(eGBufLdr));


			m_tonemapper->runCompute(vkCmd, m_gBuffers->getSize());

			// LDR barrier
			{
				VkImageMemoryBarrier ldrBarrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
				ldrBarrier.image = m_gBuffers->getColorImage(eGBufLdr);
				ldrBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
				ldrBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
				ldrBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
				ldrBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
				ldrBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
				vkCmdPipelineBarrier(vkCmd,
					VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
					VK_PIPELINE_STAGE_TRANSFER_BIT,
					0, 0, nullptr, 0, nullptr, 1, &ldrBarrier);
			}

			// Blit left half of LDR to left eye, right half to right eye
			{
				VkImage src = m_gBuffers->getColorImage(eGBufLdr);
				VkImage dst = g_openXRState.swapchainImages[swapchainImageIndex].image;
				int32_t halfWidth = static_cast<int32_t>(m_gBuffers->getSize().width / 2);
				int32_t fullHeight = static_cast<int32_t>(m_gBuffers->getSize().height);

				for (uint32_t eye = 0; eye < 2; ++eye) {
					VkImageBlit blit{};
					blit.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
					// Left eye reads from left half (x: 0..halfWidth)
					// Right eye reads from right half (x: halfWidth..width)
					blit.srcOffsets[0] = { static_cast<int32_t>(eye * halfWidth), 0, 0 };
					blit.srcOffsets[1] = { static_cast<int32_t>((eye + 1) * halfWidth), fullHeight, 1 };

					blit.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, eye, 1 };
					blit.dstOffsets[0] = { 0, 0, 0 };
					blit.dstOffsets[1] = {
						static_cast<int32_t>(g_openXRState.swapchainWidth),
						static_cast<int32_t>(g_openXRState.swapchainHeight),
						1
					};

					vkCmdBlitImage(vkCmd,
						src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
						dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
						1, &blit, VK_FILTER_LINEAR);
				}
			}

			// Final layout transitions
			{
				const uint32_t idx = swapchainImageIndex;
				VkImage swapImg = g_openXRState.swapchainImages[idx].image;

				VkImageMemoryBarrier toFinal{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
				toFinal.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
				toFinal.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
				toFinal.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
				toFinal.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
				toFinal.image = swapImg;
				toFinal.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 2 };
				toFinal.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
				toFinal.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;

				vkCmdPipelineBarrier(vkCmd,
					VK_PIPELINE_STAGE_TRANSFER_BIT,
					VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
					0, 0, nullptr, 0, nullptr, 1, &toFinal);

				g_openXRState.swapchainImageLayouts[idx] = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
			}

			{
				VkImageMemoryBarrier ldrBarrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
				ldrBarrier.image = m_gBuffers->getColorImage(eGBufLdr);
				ldrBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
				ldrBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
				ldrBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
				ldrBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
				ldrBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
				vkCmdPipelineBarrier(vkCmd,
					VK_PIPELINE_STAGE_TRANSFER_BIT,
					VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
					0, 0, nullptr, 0, nullptr, 1, &ldrBarrier);
			}

			// End command buffer BEFORE releasing swapchain
			vkEndCommandBuffer(vkCmd);

			// Submit GPU work and wait
			VkFence fence = VK_NULL_HANDLE;
			VkFenceCreateInfo fci{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
			vkCreateFence(m_device, &fci, nullptr, &fence);

			VkCommandBufferSubmitInfo cmdInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
			cmdInfo.commandBuffer = vkCmd;
			VkSubmitInfo2 submit2{ VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
			submit2.commandBufferInfoCount = 1;
			submit2.pCommandBufferInfos = &cmdInfo;

			vkQueueSubmit2(m_app->getQueue(0).queue, 1, &submit2, fence);
			vkWaitForFences(m_device, 1, &fence, VK_TRUE, UINT64_MAX);
			vkDestroyFence(m_device, fence, nullptr);

			// Release swapchain AFTER GPU finishes
			{
				XrSwapchainImageReleaseInfo releaseInfo{ XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO };
				XR_CHECK(xrReleaseSwapchainImage(g_openXRState.swapchain, &releaseInfo));
			}

			// Build projection layer views
			projectionViews.resize(2);
			for (uint32_t eye = 0; eye < 2; ++eye)
			{
				XrCompositionLayerProjectionView pv{ XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW };
				pv.next = nullptr;
				pv.pose = views[eye].pose;
				pv.fov = views[eye].fov;

				pv.subImage.swapchain = g_openXRState.swapchain;
				pv.subImage.imageRect.offset = { 0, 0 };
				pv.subImage.imageRect.extent = {
				  static_cast<int32_t>(g_openXRState.swapchainWidth),
				  static_cast<int32_t>(g_openXRState.swapchainHeight)
				};
				pv.subImage.imageArrayIndex = eye;

				projectionViews[eye] = pv;
			}

			// End XR frame with projection layer
			XrCompositionLayerProjection layer{ XR_TYPE_COMPOSITION_LAYER_PROJECTION };
			layer.next = nullptr;
			layer.layerFlags = 0;
			layer.space = g_openXRState.referenceSpace;
			layer.viewCount = 2;
			layer.views = projectionViews.data();

			XrCompositionLayerBaseHeader* layers[] = {
			  reinterpret_cast<XrCompositionLayerBaseHeader*>(&layer)
			};

			XrFrameEndInfo endInfo{ XR_TYPE_FRAME_END_INFO };
			endInfo.next = nullptr;
			endInfo.displayTime = xrFrameState.predictedDisplayTime;
			endInfo.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
			endInfo.layerCount = 1;
			endInfo.layers = layers;

			XR_CHECK(xrEndFrame(g_openXRState.session, &endInfo));
		}

		// non-XR Raytracing render
		void onRenderNonVR(VkCommandBuffer /*cmd*/)
		{
			if (!m_scene->valid())
				return;
			// Update the frame only if the scene is valid
			if (!updateFrame())
				return;

			// Using local command buffer for the frame
			const CommandFrame& commandFrame = m_commandFrames[m_app->getFrameCycleIndex()];
			VkCommandBuffer cmd = commandFrame.cmdBuffer[0];
			vkResetCommandPool(m_device, commandFrame.cmdPool, 0);
			VkCommandBufferBeginInfo begin_info{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, 0, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
			vkBeginCommandBuffer(cmd, &begin_info);

			auto scope_dbg = m_dutil->DBG_SCOPE(cmd);

			std::cout << "Screen resolution: " << m_viewSize.x << "x" << m_viewSize.y << std::endl;

			// Get camera info
			float view_aspect_ratio = (m_viewSize.x * 0.5) / m_viewSize.y;
			float eyeOffset = 2.04f; // Adjust this value as needed

			glm::vec3 eyeMid, eyeLeft, eyeRight;
			glm::vec3 center = CameraManip.getCenter();
			glm::vec3 up = CameraManip.getUp();

			glm::vec2 clip;
			CameraManip.setFov(90);
			eyeMid = CameraManip.getEye();
			eyeLeft = eyeMid - glm::vec3(eyeOffset, 0.0f, 0.0f);
			CameraManip.setLookat(eyeLeft, center, up);
			clip = CameraManip.getClipPlanes();
			m_frameInfo.view = CameraManip.getMatrix();
			m_frameInfo.proj = glm::perspectiveRH_ZO(glm::radians(CameraManip.getFov()), view_aspect_ratio, clip.x, clip.y);
			m_frameInfo.proj[1][1] *= -1;
			m_frameInfo.camPos = glm::vec4(eyeLeft, 0.0f);

			eyeRight = eyeMid + glm::vec3(eyeOffset, 0.0f, 0.0f);
			CameraManip.setLookat(eyeRight, center, up);
			clip = CameraManip.getClipPlanes();
			m_frameInfo.view2 = CameraManip.getMatrix();
			m_frameInfo.proj2 = glm::perspectiveRH_ZO(glm::radians(CameraManip.getFov()), view_aspect_ratio, clip.x, clip.y);
			m_frameInfo.proj2[1][1] *= -1;
			m_frameInfo.camPos2 = glm::vec4(eyeRight, 0.0f);

			m_frameInfo.envRotation = m_settings.envRotation;
			m_frameInfo.clearColor = m_settings.clearColor;

			CameraManip.setLookat(eyeMid, center, up);

			vkCmdUpdateBuffer(cmd, m_bFrameInfo.buffer, 0, sizeof(FrameInfo), &m_frameInfo);

			// Push constant
			m_pushConst.maxDepth = m_settings.maxDepth;
			m_pushConst.maxSamples = m_settings.maxSamples;
			m_pushConst.frame = m_frame;
			m_pushConst.middleRadius = m_middleRadius;

			raytraceScene(cmd);

#ifdef NVP_SUPPORTS_OPTIX7
			// #OPTIX_D
			if (needToDenoise())
			{
				// Submit raytracing and signal
				copyImagesToCuda(cmd);
				vkEndCommandBuffer(cmd); // Need to end the command buffer to submit the semaphore

				// Prepare the signal semaphore for the OptiX denoiser
				VkSemaphoreSubmitInfoKHR signal_semaphore{
					.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO_KHR,
					.semaphore = m_denoiser->getTLSemaphore(),
					.value = ++m_fenceValue, // Increment for signaling
					.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT_KHR,
				};

				VkCommandBufferSubmitInfoKHR cmd_buf_info{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO_KHR, 0, cmd };

				VkSubmitInfo2KHR submits{
					.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2_KHR,
					.commandBufferInfoCount = 1,
					.pCommandBufferInfos = &cmd_buf_info,
					.signalSemaphoreInfoCount = 1,
					.pSignalSemaphoreInfos = &signal_semaphore,
				};

				// Submit rendering and signal when done
				vkQueueSubmit2(m_app->getQueue(0).queue, 1, &submits, {});

				// #OPTIX_D
				// Denoiser waits for signal (Vulkan) and submit (Cuda) new one when done
				denoiseImage();

				// #OPTIX_D
				// Adding a wait semaphore to the application, such that the frame command buffer,
				// will wait for the end of the denoised image before executing the command buffer.
				VkSemaphoreSubmitInfo wait_semaphore{
					.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO_KHR,
					.semaphore = m_denoiser->getTLSemaphore(),
					.value = m_fenceValue,
					.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
				};
				m_app->addWaitSemaphore(wait_semaphore);

				// #OPTIX_D
				// Continue rendering pipeline (using the second command buffer)
				cmd = commandFrame.cmdBuffer[1];

				VkCommandBufferBeginInfo begin_info{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, 0, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
				vkBeginCommandBuffer(cmd, &begin_info);
				copyCudaImagesToVulkan(cmd);
			}
#endif

			// Apply tonemapper - take GBuffer-X and output to GBuffer-0
			m_tonemapper->runCompute(cmd, m_gBuffers->getSize());

			// End of the first or second command buffer
			vkEndCommandBuffer(cmd);
			VkCommandBufferSubmitInfo submit_info{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO_KHR };
			submit_info.commandBuffer = cmd;
			m_app->prependCommandBuffer(submit_info); // Prepend to the frame command buffer
		}

		void onRender(VkCommandBuffer cmd) override
		{
			if (m_enableXR)
			{
				onRenderVR(cmd);
			}
			else
			{
				onRenderNonVR(cmd);
			}
		}

	private:
		void createScene(const std::string& filename)
		{
			m_scene->load(filename);
			nvvkhl::setCamera(filename, m_scene->getRenderCameras(), m_scene->getSceneBounds()); // Camera auto-scene-fitting
			g_elemCamera->setSceneRadius(m_scene->getSceneBounds().radius());                    // Navigation help

			{ // Create the Vulkan side of the scene
				auto cmd = m_app->createTempCmdBuffer();
				m_sceneVk->create(cmd, *m_scene);
				m_sceneRtx->create(cmd, *m_scene, *m_sceneVk, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR); // Create BLAS / TLAS
				m_app->submitAndWaitTempCmdBuffer(cmd);

				m_picker->setTlas(m_sceneRtx->tlas());
			}

			m_allNodes = m_scene->getShadedNodes(nvh::gltf::Scene::PipelineType::eRasterAll);
			m_solidMatNodes = m_scene->getShadedNodes(nvh::gltf::Scene::PipelineType::eRasterSolid);
			m_blendMatNodes = m_scene->getShadedNodes(nvh::gltf::Scene::PipelineType::eRasterBlend);

			// Descriptor Set and Pipelines
			createSceneSet();
			createRtxSet();
			createRtxPipeline(); // must recreate due to texture changes
			writeSceneSet();
			writeRtxSet();
		}

		void createGbuffers(const glm::vec2& size)
		{
			static auto depth_format = nvvk::findDepthFormat(m_app->getPhysicalDevice()); // Not all depth are supported

			m_viewSize = size;
			VkExtent2D vk_size{ static_cast<uint32_t>(m_viewSize.x), static_cast<uint32_t>(m_viewSize.y) };

			// Four GBuffers: RGBA8 and 4x RGBA32F(final,albedo,normal, denoised), rendering to RGBA32F and tone mapped to RGBA8
			std::vector<VkFormat> color_buffers = {
				VK_FORMAT_B8G8R8A8_UNORM,      // LDR
				VK_FORMAT_R32G32B32A32_SFLOAT, // Result
				VK_FORMAT_R16G16B16A16_SFLOAT,  // Albedo
				VK_FORMAT_R16G16B16A16_SFLOAT,  // Normal
				VK_FORMAT_R16G16B16A16_SFLOAT,  // Depth
				VK_FORMAT_R16G16B16A16_SFLOAT,  // Disparity
				VK_FORMAT_R16G16B16A16_SFLOAT,  // Denoised
				//VK_FORMAT_R32G32B32A32_SFLOAT, // Albedo
				//VK_FORMAT_R32G32B32A32_SFLOAT, // Normal
				//VK_FORMAT_R32G32B32A32_SFLOAT, // Depth
				//VK_FORMAT_R32G32B32A32_SFLOAT, // Disparity
				//VK_FORMAT_R32G32B32A32_SFLOAT, // Denoised
			};

			// Creation of the GBuffers
			m_gBuffers = std::make_unique<nvvkhl::GBuffer>(m_device, m_alloc.get(), vk_size, color_buffers, depth_format);

#ifdef NVP_SUPPORTS_OPTIX7
			// Only allocate denoiser buffers when denoising is actually enabled
			//if (m_settings.denoiseApply && !m_enableXR)
			if (m_settings.denoiseApply)

			{
				m_denoiser->allocateBuffers(vk_size);
			};
#endif

			// Indicate the renderer to reset its frame
			resetFrame();
		}

#define GRID_SIZE 16
		inline VkExtent2D getGridSize(const VkExtent2D& size)
		{
			return VkExtent2D{ (size.width + (GRID_SIZE - 1)) / GRID_SIZE, (size.height + (GRID_SIZE - 1)) / GRID_SIZE };
		}

		// Create all Vulkan buffer data
		void createVulkanBuffers()
		{
			auto* cmd = m_app->createTempCmdBuffer();

			// Create the buffer of the current frame, changing at each frame
			m_bFrameInfo = m_alloc->createBuffer(sizeof(FrameInfo), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

			m_dutil->DBG_NAME(m_bFrameInfo.buffer);
			m_app->submitAndWaitTempCmdBuffer(cmd);
		}

		void createRtxSet()
		{
			auto& d = m_rtxSet;
			d->deinit();
			d->init(m_device);

			// This descriptor set, holds the top level acceleration structure and the output image
			d->addBinding(RtxBindings::eTlas, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_ALL);
			d->addBinding(RtxBindings::eOutImage, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
			// #OPTIX_D
			d->addBinding(RtxBindings::eOutAlbedo, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
			d->addBinding(RtxBindings::eOutNormal, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
			d->addBinding(RtxBindings::eOutDepth, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
			d->addBinding(RtxBindings::eOutDisparity, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);

			d->initLayout();
			d->initPool(1);
			m_dutil->DBG_NAME(d->getLayout());
			m_dutil->DBG_NAME(d->getSet());
		}

		void createSceneSet()
		{
			auto& d = m_sceneSet;
			d->deinit();
			d->init(m_device);

			// This descriptor set, holds the top level acceleration structure and the output image
			d->addBinding(SceneBindings::eFrameInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);
			d->addBinding(SceneBindings::eSceneDesc, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
			d->addBinding(SceneBindings::eTextures, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, m_sceneVk->nbTextures(), VK_SHADER_STAGE_ALL);
			d->initLayout();
			d->initPool(1);
			m_dutil->DBG_NAME(d->getLayout());
			m_dutil->DBG_NAME(d->getSet());
		}

		//--------------------------------------------------------------------------------------------------
		// Pipeline for the ray tracer: all shaders, raygen, chit, miss
		void createRtxPipeline()
		{
			auto& p = m_rtxPipe;
			p.destroy(m_device);
			p.plines.resize(1);

			// Creating all shaders
			enum StageIndices
			{
				eRaygen,
				eMiss,
				eMissGbuf, // #OPTIX_D
				eClosestHit,
				eAnyHit,
				eClosestHitGbuf, // #OPTIX_D
				eShaderGroupCount
			};
			std::array<VkPipelineShaderStageCreateInfo, eShaderGroupCount> stages{};
			VkPipelineShaderStageCreateInfo stage{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
			stage.pName = "main"; // All the same entry point
			// Raygen
			stage.module = nvvk::createShaderModule(m_device, pathtrace_rgen, sizeof(pathtrace_rgen));
			stage.stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
			stages[eRaygen] = stage;
			m_dutil->setObjectName(stage.module, "Raygen");
			// Miss
			stage.module = nvvk::createShaderModule(m_device, pathtrace_rmiss, sizeof(pathtrace_rmiss));
			stage.stage = VK_SHADER_STAGE_MISS_BIT_KHR;
			stages[eMiss] = stage;
			m_dutil->setObjectName(stage.module, "Miss");
			// Hit Group - Closest Hit
			stage.module = nvvk::createShaderModule(m_device, pathtrace_rchit, sizeof(pathtrace_rchit));
			stage.stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
			stages[eClosestHit] = stage;
			m_dutil->setObjectName(stage.module, "Closest Hit");
			// AnyHit
			stage.module = nvvk::createShaderModule(m_device, pathtrace_rahit, sizeof(pathtrace_rahit));
			stage.stage = VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
			stages[eAnyHit] = stage;
			m_dutil->setObjectName(stage.module, "Any Hit");

			// Miss G-Buffers
			stage.module = nvvk::createShaderModule(m_device, gbuffers_rmiss, sizeof(gbuffers_rmiss));
			stage.stage = VK_SHADER_STAGE_MISS_BIT_KHR;
			stages[eMissGbuf] = stage;
			// Hit Group - Closest Hit
			stage.module = nvvk::createShaderModule(m_device, gbuffers_rchit, sizeof(gbuffers_rchit));
			stage.stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
			stages[eClosestHitGbuf] = stage;

			// Shader groups
			VkRayTracingShaderGroupCreateInfoKHR group{ VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR };
			group.anyHitShader = VK_SHADER_UNUSED_KHR;
			group.closestHitShader = VK_SHADER_UNUSED_KHR;
			group.generalShader = VK_SHADER_UNUSED_KHR;
			group.intersectionShader = VK_SHADER_UNUSED_KHR;

			std::vector<VkRayTracingShaderGroupCreateInfoKHR> shader_groups;
			// Raygen
			group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
			group.generalShader = eRaygen;
			shader_groups.push_back(group);

			// Miss
			group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
			group.generalShader = eMiss;
			shader_groups.push_back(group);

			// #OPTIX_D
			// Miss - G-Buf
			group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
			group.generalShader = eMissGbuf;
			shader_groups.push_back(group);

			// Hit Group-0
			group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
			group.generalShader = VK_SHADER_UNUSED_KHR;
			group.closestHitShader = eClosestHit;
			group.anyHitShader = eAnyHit;
			shader_groups.push_back(group);

			// #OPTIX_D
			// Hit Group-1
			group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
			group.generalShader = VK_SHADER_UNUSED_KHR;
			group.closestHitShader = eClosestHitGbuf;
			group.anyHitShader = VK_SHADER_UNUSED_KHR;
			shader_groups.push_back(group);

			// Push constant: we want to be able to update constants used by the shaders
			VkPushConstantRange push_constant{ VK_SHADER_STAGE_ALL, 0, sizeof(PushConstant) };

			VkPipelineLayoutCreateInfo pipeline_layout_create_info{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
			pipeline_layout_create_info.pushConstantRangeCount = 1;
			pipeline_layout_create_info.pPushConstantRanges = &push_constant;

			// Descriptor sets: one specific to ray tracing, and one shared with the rasterization pipeline
			std::vector<VkDescriptorSetLayout> rt_desc_set_layouts = { m_rtxSet->getLayout(), m_sceneSet->getLayout(),
																	  m_hdrEnv->getDescriptorSetLayout() };
			pipeline_layout_create_info.setLayoutCount = static_cast<uint32_t>(rt_desc_set_layouts.size());
			pipeline_layout_create_info.pSetLayouts = rt_desc_set_layouts.data();
			vkCreatePipelineLayout(m_device, &pipeline_layout_create_info, nullptr, &p.layout);
			m_dutil->DBG_NAME(p.layout);

			// Assemble the shader stages and recursion depth info into the ray tracing pipeline
			VkRayTracingPipelineCreateInfoKHR ray_pipeline_info{ VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR };
			ray_pipeline_info.stageCount = static_cast<uint32_t>(stages.size()); // Stages are shaders
			ray_pipeline_info.pStages = stages.data();
			ray_pipeline_info.groupCount = static_cast<uint32_t>(shader_groups.size());
			ray_pipeline_info.pGroups = shader_groups.data();
			ray_pipeline_info.maxPipelineRayRecursionDepth = 2; // Ray depth
			ray_pipeline_info.layout = p.layout;
			vkCreateRayTracingPipelinesKHR(m_device, {}, {}, 1, &ray_pipeline_info, nullptr, (p.plines).data());
			m_dutil->DBG_NAME(p.plines[0]);

			// Creating the SBT
			m_sbt->create(p.plines[0], ray_pipeline_info);

			// Removing temp modules
			for (auto& s : stages)
			{
				vkDestroyShaderModule(m_device, s.module, nullptr);
			}
		}

		void writeRtxSet()
		{
			if (!m_scene->valid())
			{
				return;
			}

			auto& d = m_rtxSet;

			// Write to descriptors
			VkAccelerationStructureKHR tlas = m_sceneRtx->tlas();
			VkWriteDescriptorSetAccelerationStructureKHR desc_as_info{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR };
			desc_as_info.accelerationStructureCount = 1;
			desc_as_info.pAccelerationStructures = &tlas;
			VkDescriptorImageInfo image_info{ {}, m_gBuffers->getColorImageView(eGBufResult), VK_IMAGE_LAYOUT_GENERAL };
			// #OPTIX_D
			VkDescriptorImageInfo albedo_info{ {}, m_gBuffers->getColorImageView(eGBufAlbedo), VK_IMAGE_LAYOUT_GENERAL };
			VkDescriptorImageInfo normal_info{ {}, m_gBuffers->getColorImageView(eGBufNormal), VK_IMAGE_LAYOUT_GENERAL };
			VkDescriptorImageInfo depth_info{ {}, m_gBuffers->getColorImageView(eGBufDepth), VK_IMAGE_LAYOUT_GENERAL };
			VkDescriptorImageInfo discrep_info{ {}, m_gBuffers->getColorImageView(eGBufDisparity), VK_IMAGE_LAYOUT_GENERAL };

			std::vector<VkWriteDescriptorSet> writes;
			writes.emplace_back(d->makeWrite(0, RtxBindings::eTlas, &desc_as_info));
			writes.emplace_back(d->makeWrite(0, RtxBindings::eOutImage, &image_info));
			// #OPTIX_D
			writes.emplace_back(d->makeWrite(0, RtxBindings::eOutAlbedo, &albedo_info));
			writes.emplace_back(d->makeWrite(0, RtxBindings::eOutNormal, &normal_info));
			writes.emplace_back(d->makeWrite(0, RtxBindings::eOutDepth, &depth_info));
			writes.emplace_back(d->makeWrite(0, RtxBindings::eOutDisparity, &discrep_info));

			vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
		}

		void writeSceneSet()
		{
			if (!m_scene->valid())
			{
				return;
			}

			auto& d = m_sceneSet;

			// Write to descriptors
			VkDescriptorBufferInfo dbi_unif{ m_bFrameInfo.buffer, 0, VK_WHOLE_SIZE };
			VkDescriptorBufferInfo scene_desc{ m_sceneVk->sceneDesc().buffer, 0, VK_WHOLE_SIZE };

			std::vector<VkWriteDescriptorSet> writes;
			writes.emplace_back(d->makeWrite(0, SceneBindings::eFrameInfo, &dbi_unif));
			writes.emplace_back(d->makeWrite(0, SceneBindings::eSceneDesc, &scene_desc));
			std::vector<VkDescriptorImageInfo> diit;
			for (const auto& texture : m_sceneVk->textures()) // All texture samplers
			{
				diit.emplace_back(texture.descriptor);
			}
			writes.emplace_back(d->makeWriteArray(0, SceneBindings::eTextures, diit.data()));

			vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
		}

		//--------------------------------------------------------------------------------------------------
		// If the camera matrix has changed, resets the frame. otherwise, increments frame.
		//
		bool updateFrame()
		{
			static glm::mat4 ref_cam_matrix;
			static float ref_fov{ CameraManip.getFov() };

			const auto& m = CameraManip.getMatrix();
			const auto fov = CameraManip.getFov();

			if (ref_cam_matrix != m || ref_fov != fov)
			{
				resetFrame();
				ref_cam_matrix = m;
				ref_fov = fov;
			}

			if (m_frame >= m_settings.maxFrames)
			{
				return false;
			}
			m_frame++;
			return true;
		}

		//-----------------------------------------------------------------------
		// To be call when renderer need to re-start
		//
		void resetFrame() { m_frame = -1; }

		void windowTitle()
		{
			// Window Title
			static float dirty_timer = 0.0F;
			dirty_timer += ImGui::GetIO().DeltaTime;
			if (dirty_timer > 1.0F) // Refresh every seconds
			{
				const auto& size = m_app->getViewportSize();
				std::array<char, 256> buf{};
				int ret = snprintf(buf.data(), buf.size(), "%s %dx%d | %d FPS / %.3fms | Frame %d", PROJECT_NAME,
					static_cast<int>(size.width), static_cast<int>(size.height),
					static_cast<int>(ImGui::GetIO().Framerate), 1000.F / ImGui::GetIO().Framerate, m_frame);
				glfwSetWindowTitle(m_app->getWindowHandle(), buf.data());
				dirty_timer = 0;
			}
		}

		//--------------------------------------------------------------------------------------------------
		// Send a ray under mouse coordinates, and retrieve the information
		// - Set new camera interest point on hit position
		//
		void screenPicking()
		{
			auto* tlas = m_sceneRtx->tlas();
			if (tlas == VK_NULL_HANDLE)
				return;

			ImGui::Begin("Viewport"); // ImGui, picking within "viewport"
			auto mouse_pos = ImGui::GetMousePos();
			auto main_size = ImGui::GetContentRegionAvail();
			auto corner = ImGui::GetCursorScreenPos(); // Corner of the viewport
			float aspect_ratio = main_size.x / main_size.y;
			mouse_pos = mouse_pos - corner;
			ImVec2 local_mouse_pos = mouse_pos / main_size;
			ImGui::End();

			auto* cmd = m_app->createTempCmdBuffer();

			// Finding current camera matrices
			const auto& view = CameraManip.getMatrix();
			auto proj = glm::perspectiveRH_ZO(glm::radians(CameraManip.getFov()), aspect_ratio, 0.1F, 1000.0F);
			proj[1][1] *= -1;

			// Setting up the data to do picking
			nvvk::RayPickerKHR::PickInfo pick_info;
			pick_info.pickX = local_mouse_pos.x;
			pick_info.pickY = local_mouse_pos.y;
			pick_info.modelViewInv = glm::inverse(view);
			pick_info.perspectiveInv = glm::inverse(proj);

			// Run and wait for result
			m_picker->run(cmd, pick_info);
			m_app->submitAndWaitTempCmdBuffer(cmd);

			// Retrieving picking information
			nvvk::RayPickerKHR::PickResult pr = m_picker->getResult();
			if (pr.instanceID == ~0)
			{
				LOGI("Nothing Hit\n");
				return;
			}

			if (pr.hitT <= 0.F)
			{
				LOGI("Hit Distance == 0.0\n");
				return;
			}

			// Find where the hit point is and set the interest position
			glm::vec3 world_pos = glm::vec3(pr.worldRayOrigin + pr.worldRayDirection * pr.hitT);
			glm::vec3 eye;
			glm::vec3 center;
			glm::vec3 up;
			CameraManip.getLookat(eye, center, up);
			CameraManip.setLookat(eye, world_pos, up, false);

			// Logging picking info.
			const auto& renderNode = m_scene->getRenderNodes()[pr.instanceID];
			std::string name = m_scene->getModel().nodes[renderNode.refNodeID].name;
			LOGI("Hit(%d): %s, PrimId: %d", pr.instanceCustomIndex, name.c_str(), pr.primitiveID);
			LOGI("{%3.2f, %3.2f, %3.2f}, Dist: %3.2f\n", world_pos.x, world_pos.y, world_pos.z, pr.hitT);
			LOGI("PrimitiveID: %d\n", pr.primitiveID);
		}

		void raytraceScene(VkCommandBuffer cmd)
		{
			auto scope_dbg = m_dutil->DBG_SCOPE(cmd);

			// Ray trace
			std::vector<VkDescriptorSet> desc_sets{ m_rtxSet->getSet(), m_sceneSet->getSet(), m_hdrEnv->getDescriptorSet() };
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtxPipe.plines[0]);
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtxPipe.layout, 0,
				static_cast<uint32_t>(desc_sets.size()), desc_sets.data(), 0, nullptr);

			/*
			   First pass: Dominant eye ray-tracing
			   TODO: add depth map to the push constant
			*/
			m_pushConst.passId = 0;
			vkCmdPushConstants(cmd, m_rtxPipe.layout, VK_SHADER_STAGE_ALL, 0, sizeof(PushConstant), &m_pushConst);

			const auto& regions = m_sbt->getRegions();
			const auto& size = m_gBuffers->getSize();
			vkCmdTraceRaysKHR(cmd, regions.data(), &regions[1], &regions[2], &regions[3], size.width, size.height, 1);

			/*
			  Barrier to ensure writes are visible
			*/
			VkMemoryBarrier barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
			barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
			vkCmdPipelineBarrier(cmd,
				VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
				VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
				0, 1, &barrier, 0, nullptr, 0, nullptr);

			/*
			   Second pass:
				- Non-dominant eye central ray-tracing
				- Copy pixels from the first pass (in peripheral) (TODO: ignore some side pixels)
				- TODO: Warp pixels copied on the periphery in a transition zone
				- TODO: Create a third pass: ray-tracing on the holes after copy and warping
			*/
			m_pushConst.passId = 1;
			vkCmdPushConstants(cmd, m_rtxPipe.layout, VK_SHADER_STAGE_ALL, 0, sizeof(PushConstant), &m_pushConst);
			vkCmdTraceRaysKHR(cmd, regions.data(), &regions[1], &regions[2], &regions[3], size.width, size.height, 1);

			// Making sure the rendered image is ready to be used
			{
				auto scope_dbg2 = m_dutil->scopeLabel(cmd, "barrier");

				auto image_memory_barrier =
					nvvk::makeImageMemoryBarrier(m_gBuffers->getColorImage(eGBufResult), VK_ACCESS_SHADER_READ_BIT,
						VK_ACCESS_SHADER_WRITE_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
				vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0,
					nullptr, 0, nullptr, 1, &image_memory_barrier);
			}
		}

		void createHdr(const char* filename)
		{
			m_hdrEnv = std::make_unique<HdrEnv>(m_device, m_physicalDevice, m_alloc.get());

			m_hdrEnv->loadEnvironment(filename);
		}

		// #OPTIX_D
		// Return true if the current frame need to be denoised, as we are not  denoising all frames.
		bool needToDenoise() const
		{
			if (m_settings.denoiseApply)
			{
				if (m_frame == m_settings.maxFrames)
					return true;
				if (!m_settings.denoiseFirstFrame && m_frame == 0)
					return false;
				if (m_frame % m_settings.denoiseEveryNFrames == 0)
					return true;
			}
			return false;
		}

		// #OPTIX_D
		// Will copy the Vulkan images to Cuda buffers
		void copyImagesToCuda(VkCommandBuffer cmd)
		{
#ifdef NVP_SUPPORTS_OPTIX7
			nvvk::Texture result{ m_gBuffers->getColorImage(eGBufResult), nullptr, m_gBuffers->getDescriptorImageInfo(eGBufResult) };
			nvvk::Texture albedo{ m_gBuffers->getColorImage(eGBufAlbedo), nullptr, m_gBuffers->getDescriptorImageInfo(eGBufAlbedo) };
			nvvk::Texture normal{ m_gBuffers->getColorImage(eGBufNormal), nullptr, m_gBuffers->getDescriptorImageInfo(eGBufNormal) };
			nvvk::Texture depth{ m_gBuffers->getColorImage(eGBufDepth), nullptr, m_gBuffers->getDescriptorImageInfo(eGBufDepth) };
			nvvk::Texture disparity{ m_gBuffers->getColorImage(eGBufDisparity), nullptr, m_gBuffers->getDescriptorImageInfo(eGBufDisparity) };
			m_denoiser->imageToBuffer(cmd, { result, albedo, normal, depth, disparity });
#endif // NVP_SUPPORTS_OPTIX7
		}

		// #OPTIX_D
		// Copy the denoised buffer to Vulkan image
		void copyCudaImagesToVulkan(VkCommandBuffer cmd)
		{
#ifdef NVP_SUPPORTS_OPTIX7
			nvvk::Texture denoised{ m_gBuffers->getColorImage(eGbufDenoised), nullptr, m_gBuffers->getDescriptorImageInfo(eGbufDenoised) };
			m_denoiser->bufferToImage(cmd, &denoised);
#endif // NVP_SUPPORTS_OPTIX7
		}

		// #OPTIX_D
		// Invoke the Optix denoiser
		void denoiseImage()
		{
#ifdef NVP_SUPPORTS_OPTIX7
			m_denoiser->denoiseImageBuffer(m_fenceValue, m_blendFactor);
#endif // NVP_SUPPORTS_OPTIX7
		}

		// #OPTIX_D
		// Determine which image will be displayed, the original from ray tracer or the denoised one
		bool showDenoisedImage() const
		{
			/*if (m_enableXR)
				return false;*/
			return m_settings.denoiseApply && ((m_frame >= m_settings.denoiseEveryNFrames) || m_settings.denoiseFirstFrame || (m_frame >= m_settings.maxFrames));
		}

		void createCommandBuffers()
		{
			// Max 3 frames in flight
			for (uint32_t i = 0; i < 3; i++)
			{
				CommandFrame* cf = &m_commandFrames[i];
				{
					VkCommandPoolCreateInfo info = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
													.flags = 0,
													.queueFamilyIndex = m_app->getQueue(0).familyIndex };
					NVVK_CHECK(vkCreateCommandPool(m_device, &info, nullptr, &cf->cmdPool));
					m_dutil->setObjectName(cf->cmdPool, "Pool" + std::to_string(i));
				}
				{
					VkCommandBufferAllocateInfo info = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
														.commandPool = cf->cmdPool,
														.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
														.commandBufferCount = 2 };
					NVVK_CHECK(vkAllocateCommandBuffers(m_device, &info, cf->cmdBuffer));
					m_dutil->setObjectName(cf->cmdBuffer[0], fmt::format("Cmd[{}][0]", i));
					m_dutil->setObjectName(cf->cmdBuffer[1], fmt::format("Cmd[{}][1]", i));
				}
			}
		}

		void destroyResources()
		{
			m_alloc->destroy(m_bFrameInfo);

			for (auto& f : m_commandFrames)
			{
				vkFreeCommandBuffers(m_device, f.cmdPool, 2, f.cmdBuffer);
				vkDestroyCommandPool(m_device, f.cmdPool, nullptr);
			}
			m_gBuffers.reset();

			m_rasterPipe.destroy(m_device);
			m_rtxPipe.destroy(m_device);
			m_rtxSet->deinit();
			m_sceneSet->deinit();
			m_sbt->destroy();
			m_picker->destroy();
#ifdef NVP_SUPPORTS_OPTIX7
			m_denoiser->destroy();
#endif
		}

		//--------------------------------------------------------------------------------------------------
		//
		//
		nvvkhl::Application* m_app{ nullptr };
		std::unique_ptr<nvvk::DebugUtil> m_dutil;
		std::unique_ptr<AllocVma> m_alloc;

		glm::vec2 m_viewSize = { 1, 1 };
		VkClearColorValue m_clearColor = { {0.3F, 0.3F, 0.3F, 1.0F} }; // Clear color
		VkDevice m_device = VK_NULL_HANDLE;                          // Convenient
		VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;          // Convenient
		std::unique_ptr<nvvkhl::GBuffer> m_gBuffers;                 // G-Buffers: color + depth
		std::unique_ptr<nvvk::DescriptorSetContainer> m_rtxSet;      // Descriptor set
		std::unique_ptr<nvvk::DescriptorSetContainer> m_sceneSet;    // Descriptor set

		// Resources
		nvvk::Buffer m_bFrameInfo;

		// Pipeline
		PushConstant m_pushConst{}; // Information sent to the shader
		PipelineContainer m_rasterPipe;
		PipelineContainer m_rtxPipe;
		int m_frame{ -1 };
		FrameInfo m_frameInfo;

		std::unique_ptr<nvh::gltf::Scene> m_scene;
		std::unique_ptr<SceneVk> m_sceneVk;
		std::unique_ptr<SceneRtx> m_sceneRtx;
		std::unique_ptr<TonemapperPostProcess> m_tonemapper;
		std::unique_ptr<nvvk::SBTWrapper> m_sbt;      // Shading binding table wrapper
		std::unique_ptr<nvvk::RayPickerKHR> m_picker; // For ray picking info
		std::unique_ptr<HdrEnv> m_hdrEnv;

		// For rendering all nodes
		std::vector<uint32_t> m_solidMatNodes;
		std::vector<uint32_t> m_blendMatNodes;
		std::vector<uint32_t> m_allNodes;

#ifdef NVP_SUPPORTS_OPTIX7
		std::unique_ptr<DenoiserOptix> m_denoiser;
		uint64_t m_fenceValue{ 0U };
#endif // NVP_SUPPORTS_OPTIX7
		float m_blendFactor = 0.0f;
		float m_middleRadius = 0.6f;
		float m_xrEyeSeparation = 0.063f;

		// Command buffers for rendering
		struct CommandFrame
		{
			VkCommandPool cmdPool = VK_NULL_HANDLE;
			VkCommandBuffer cmdBuffer[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
		};
		std::array<CommandFrame, 3> m_commandFrames;
	};

} // namespace nvvkhl


/////////////////////////////////////////
auto main(int argc, char** argv) -> int
{
	nvvkhl::ApplicationCreateInfo spec;
	spec.name = PROJECT_NAME " Example";
	spec.vSync = true;

	nvvk::ContextCreateInfo vkSetup;
	vkSetup.apiMajor = 1;
	vkSetup.apiMinor = 2;
	// vkSetup.apiMinor = 3;

	std::cout << "XR_CURRENT_API_VERSION: "
		<< XR_VERSION_MAJOR(XR_CURRENT_API_VERSION) << "."
		<< XR_VERSION_MINOR(XR_CURRENT_API_VERSION) << "."
		<< XR_VERSION_PATCH(XR_CURRENT_API_VERSION) << std::endl;

	vkSetup.addDeviceExtension(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
	// #VKRay: Activate the ray tracing extension
	VkPhysicalDeviceAccelerationStructureFeaturesKHR accel_feature{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR };
	vkSetup.addDeviceExtension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, false, &accel_feature); // To build acceleration structures
	VkPhysicalDeviceRayTracingPipelineFeaturesKHR rt_pipeline_feature{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR };
	vkSetup.addDeviceExtension(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, false, &rt_pipeline_feature); // To use vkCmdTraceRaysKHR
	vkSetup.addDeviceExtension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);                          // Required by ray tracing pipeline
	VkPhysicalDeviceRayQueryFeaturesKHR ray_query_features{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR };
	vkSetup.addDeviceExtension(VK_KHR_RAY_QUERY_EXTENSION_NAME, false, &ray_query_features); // Used for picking
	vkSetup.addDeviceExtension(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);

	// #OPTIX_D
	// Semaphores - interop Vulkan/Cuda
	vkSetup.addDeviceExtension(VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
	vkSetup.addDeviceExtension(VK_KHR_EXTERNAL_FENCE_EXTENSION_NAME);
#ifdef WIN32
	vkSetup.addDeviceExtension(VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
	vkSetup.addDeviceExtension(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
	vkSetup.addDeviceExtension(VK_KHR_EXTERNAL_FENCE_WIN32_EXTENSION_NAME);
#else
	vkSetup.addDeviceExtension(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
	vkSetup.addDeviceExtension(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
	vkSetup.addDeviceExtension(VK_KHR_EXTERNAL_FENCE_FD_EXTENSION_NAME);
#endif

	// Synchronization (mix of timeline and binary semaphores)
	vkSetup.addDeviceExtension(VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME, false);

	// Buffer - interop
	vkSetup.addDeviceExtension(VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME);
	vkSetup.addDeviceExtension(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);

	// Display extension
	vkSetup.deviceExtensions.emplace_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
	vkSetup.instanceExtensions.emplace_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
	nvvkhl::addSurfaceExtensions(vkSetup.instanceExtensions);

	/*std::cout << "\n=== Setting SteamVR Runtime Path ===" << std::endl;
	_putenv_s("XR_RUNTIME_JSON", "F:\\Apps\\Steam\\steamapps\\common\\SteamVR\\steamxr_win64.json");*/

	// Creating the Vulkan context
	// auto context = createCompatibleVulkanContext();
	auto context = std::make_shared<nvvk::Context>();
	context->init(vkSetup);
	// auto m_context = std::make_shared<nvvk::Context>();
	// m_context->init(vkSetup);

	// Abort early if no valid Vulkan physical device/device or if it's a CPU device (llvmpipe)
	if (context->m_physicalDevice == VK_NULL_HANDLE || context->m_device == VK_NULL_HANDLE)
	{
		std::cerr << "No compatible Vulkan device found. This sample requires a GPU with Vulkan ray tracing and external memory/semaphore support." << std::endl;
		return 1;
	}

	VkPhysicalDeviceProperties physProps{};
	vkGetPhysicalDeviceProperties(context->m_physicalDevice, &physProps);
	if (physProps.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU)
	{
		std::cerr << "Selected Vulkan device is a CPU implementation (" << physProps.deviceName
			<< "). A discrete/integrated GPU with Vulkan ray tracing is required." << std::endl;
		return 1;
	}

	// At the very beginning of main(), add:
	std::cout << "=== Starting VK_DENOISE_VR ===" << std::endl;
	std::cout << "Environment check:" << std::endl;
	const char* xr_runtime = std::getenv("XR_RUNTIME_JSON");
	if (xr_runtime)
	{
		std::cout << "  XR_RUNTIME_JSON: " << xr_runtime << std::endl;
	}
	else
	{
		std::cout << "  XR_RUNTIME_JSON: NOT SET" << std::endl;
	}

	/*
	  OpenXR
	*/

	// 1. Create compatible Vulkan context
	std::cout << "\n=== Attempting OpenXR Initialization ===" << std::endl;


	// Keep only:
	m_enableXR = initializeOpenXR(context);

	if (m_enableXR)
	{
		try
		{
			getSystemOpenXR();
			createOpenXRSwapchain();
			std::cout << "\n=== OpenXR VR Mode Enabled ===" << std::endl;
			std::cout << "Swapchain created with " << g_openXRState.swapchainImages.size() << " images" << std::endl;
			std::cout << "Resolution: " << systemProperties.graphicsProperties.maxSwapchainImageWidth
				<< "x" << systemProperties.graphicsProperties.maxSwapchainImageHeight << std::endl;

		}
		catch (const std::exception& e)
		{
			std::cerr << "OpenXR post-init exception: " << e.what() << std::endl;
			std::cerr << "Falling back to desktop mode." << std::endl;
			m_enableXR = false;
		}
	}
	else
	{
		std::cout << "\n=== OpenXR initialization failed, falling back to Desktop Mode ===" << std::endl;
	}


	// Debug output to verify mode
	std::cout << "\n=== Final Mode: " << (m_enableXR ? "VR MODE" : "DESKTOP MODE") << " ===" << std::endl;

	// Application Vulkan setup ---------------------------
	spec.instance = context->m_instance;
	spec.device = context->m_device;
	spec.physicalDevice = context->m_physicalDevice;
	spec.queues.push_back({ context->m_queueGCT.familyIndex,
						   context->m_queueGCT.queueIndex,
						   context->m_queueGCT.queue });

	// Create the application
	auto app = std::make_unique<nvvkhl::Application>(spec);

	g_elemBenchmark = std::make_shared<nvvkhl::ElementBenchmarkParameters>(argc, argv); // Benchmarking
	g_elemCamera = std::make_shared<nvvkhl::ElementCamera>();                           // Create the camera to be used
	auto optixDenoiser = std::make_shared<nvvkhl::OptixDenoiserEngine>();               // Create application elements

	app->addElement(g_elemCamera);
	app->addElement(g_elemBenchmark);
	app->addElement(optixDenoiser);
	app->addElement(std::make_shared<nvvkhl::ElementDefaultMenu>()); // Menu / Quit

	// Search paths
	std::vector<std::string> default_search_paths = { ".", "..", "../..", "../../.." };

	// Load scene
	//std::string scn_file = nvh::findFile(R"(media/cornellBox.gltf)", default_search_paths, true);
	std::string scn_file = nvh::findFile(R"(media/sponza/glTF/Sponza.gltf)", default_search_paths, true);
	optixDenoiser->onFileDrop(scn_file.c_str());

	CameraManip.setLookat(
		glm::vec3(0.0f, 1.6f, 0.0f),   // eye:    center of atrium, standing eye height
		glm::vec3(10.0f, 1.6f, 0.0f),  // center: looking down the long axis (+X)
		glm::vec3(0.0f, 1.0f, 0.0f),   // up
		true                            // instant (no animation)
	);
	CameraManip.setFov(90.0f);

	// Load HDR
	std::string hdr_file = nvh::findFile(R"(media/hdr/kloppenheim_06_puresky_1k.hdr)", default_search_paths, true);
	//std::string hdr_file = nvh::findFile(R"(media/hdr/kloppenheim_06_4k.hdr)", default_search_paths, true);
	//std::string hdr_file = nvh::findFile(R"(media/hdr/kloppenheim_06_4k.hdr)", default_search_paths, true);

	optixDenoiser->onFileDrop(hdr_file.c_str());

	// Run as fast as possible
	app->setVsync(false);

	app->run();

	// Cleanup OpenXR resources
	if (m_enableXR)
	{
		g_openXRState.cleanup();
	}
	optixDenoiser.reset();
	app.reset();

	return 0;
}
