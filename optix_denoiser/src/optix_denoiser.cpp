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

#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <random>
#include <algorithm>

#ifdef _WIN32
#include <Windows.h>
#include <cstdlib>

// CPU-side optical constants used by raygen shader
static constexpr float HOST_MAX_COMFORTABLE_PARALLAX_ANGLE = 1.5f; // degrees
static constexpr float HOST_VIEWER_DISTANCE = 0.5f; // meters

static const char* sceneNames[] = { "Sponza", "Chess", "City" };
static const char* sceneFiles[] = {
	"media/sponza/glTF/Sponza.gltf",
	//"media/scenes/ABeautifulGame/glTF/ABeautifulGame.gltf"
	"media/scenes/ABeautifulGameCopy/Untitled.gltf",
	"media/scenes/City/scene.gltf"
};
constexpr int sceneCount = sizeof(sceneNames) / sizeof(sceneNames[0]);

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



struct HeadsetDisplayInfo {
	std::string headsetName;
	float displayWidth = 0.0f;  // Single eye display width in meters
	float displayHeight = 0.0f;
	float pixelWidth = 0.0f;
	float pixelHeight = 0.0f;
	float ipd = 0.063f;
	float fovDegrees = 88.0f;
	float fovRadians = glm::radians(97.0f);
};


OpenXRState g_openXRState;
HeadsetDisplayInfo g_headsetDisplayInfo;
XrView g_xrViews[2] = { { XR_TYPE_VIEW }, { XR_TYPE_VIEW } };

bool g_enableXR = false;
bool g_useVulkan2 = false;

std::shared_ptr<nvvkhl::ElementCamera> g_elemCamera;
std::shared_ptr<nvvkhl::ElementBenchmarkParameters> g_elemBenchmark;

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



void updateHeadsetDisplayInfo()
{
	// Fill runtime-provided metadata
	g_headsetDisplayInfo.headsetName = std::string(systemProperties.systemName);

	// Default pixel size from swapchain if available
	g_headsetDisplayInfo.pixelWidth = static_cast<float>(g_openXRState.swapchainWidth);
	g_headsetDisplayInfo.pixelHeight = static_cast<float>(g_openXRState.swapchainHeight);

	// Try to get per-eye recommended image rect from view configuration (preferred)
	uint32_t viewCount = 0;
	xrEnumerateViewConfigurationViews(g_openXRState.instance, g_openXRState.systemId,
		XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, 0, &viewCount, nullptr);

	if (viewCount >= 2)
	{
		std::vector<XrViewConfigurationView> viewsConfig(viewCount, { XR_TYPE_VIEW_CONFIGURATION_VIEW });
		xrEnumerateViewConfigurationViews(g_openXRState.instance, g_openXRState.systemId,
			XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, viewCount, &viewCount, viewsConfig.data());

		// Use first view's recommended image rect (per-eye pixels)
		g_headsetDisplayInfo.pixelWidth = static_cast<float>(viewsConfig[0].recommendedImageRectWidth);
		g_headsetDisplayInfo.pixelHeight = static_cast<float>(viewsConfig[0].recommendedImageRectHeight);

		// Calculate FOV from view configuration
		float leftFovH = glm::degrees(g_xrViews[0].fov.angleRight - g_xrViews[0].fov.angleLeft);
		float rightFovH = glm::degrees(g_xrViews[1].fov.angleRight - g_xrViews[1].fov.angleLeft);
		g_headsetDisplayInfo.fovDegrees = (leftFovH + rightFovH) * 0.5f;
		g_headsetDisplayInfo.fovRadians = glm::radians(g_headsetDisplayInfo.fovDegrees);
	}

	// Meta Quest 3S physical specifications
	// Based on published specs: single LCD panel ~1832x1920 per eye
	// Physical dimensions estimated from Quest 2/3 family
	const std::string hn = g_headsetDisplayInfo.headsetName;

	if (hn.find("Quest 3") != std::string::npos ||
		hn.find("Quest 3S") != std::string::npos ||
		hn.find("Meta Quest 3") != std::string::npos) {

		// Meta Quest 3S specifications:
		// - Display: Single fast-switch LCD per eye
		// - Resolution: 1832 x 1920 pixels per eye
		// - Physical panel size: ~2.48 inches diagonal per eye (~63mm)
		// - Estimated physical width: ~55mm (0.055m), height: ~57mm (0.057m)
		// - Fixed focus distance: ~1.3 meters (same as Quest 2/3 family)
		// - Typical IPD range: 58-68mm (hardware adjustable)

		g_headsetDisplayInfo.displayWidth = 0.055f;   // meters (estimated physical panel width)
		g_headsetDisplayInfo.displayHeight = 0.057f;  // meters (estimated physical panel height)

		std::cout << "Detected Meta Quest 3/3S headset" << std::endl;
		std::cout << "  Physical panel: " << g_headsetDisplayInfo.displayWidth * 1000.0f
			<< "mm x " << g_headsetDisplayInfo.displayHeight * 1000.0f << "mm" << std::endl;
	}
	else if (hn.find("Quest 2") != std::string::npos || hn.find("Oculus Quest 2") != std::string::npos) {
		// Quest 2: similar panel size
		g_headsetDisplayInfo.displayWidth = 0.053f;
		g_headsetDisplayInfo.displayHeight = 0.058f;
		std::cout << "Detected Meta Quest 2 headset" << std::endl;
	}
	else {
		// Fallback: estimate from typical VR panel PPI (~600-800 PPI)
		float estimatedPPI = 700.0f; // Conservative estimate for modern VR
		float diagonalInches = std::sqrt(
			g_headsetDisplayInfo.pixelWidth * g_headsetDisplayInfo.pixelWidth +
			g_headsetDisplayInfo.pixelHeight * g_headsetDisplayInfo.pixelHeight
		) / estimatedPPI;

		float aspectRatio = g_headsetDisplayInfo.pixelWidth / g_headsetDisplayInfo.pixelHeight;
		// diagonal² = width² + height², and width = aspectRatio * height
		float heightInches = diagonalInches / std::sqrt(1.0f + aspectRatio * aspectRatio);
		float widthInches = heightInches * aspectRatio;

		g_headsetDisplayInfo.displayWidth = widthInches * 0.0254f;   // Convert to meters
		g_headsetDisplayInfo.displayHeight = heightInches * 0.0254f; // Convert to meters

		std::cout << "Unknown headset: " << hn << std::endl;
		std::cout << "  Estimated panel: " << g_headsetDisplayInfo.displayWidth * 1000.0f
			<< "mm x " << g_headsetDisplayInfo.displayHeight * 1000.0f << "mm"
			<< " (at " << estimatedPPI << " PPI)" << std::endl;
	}

	// Log all headset info
	std::cout << "Headset Display Info:" << std::endl;
	std::cout << "  Name: " << g_headsetDisplayInfo.headsetName << std::endl;
	std::cout << "  Resolution: " << g_headsetDisplayInfo.pixelWidth
		<< "x" << g_headsetDisplayInfo.pixelHeight << std::endl;
	std::cout << "  FOV: " << g_headsetDisplayInfo.fovDegrees << " degrees" << std::endl;

	if (g_headsetDisplayInfo.displayWidth > 0.0f) {
		float ppi = g_headsetDisplayInfo.pixelWidth / (g_headsetDisplayInfo.displayWidth / 0.0254f);
		std::cout << "  PPI: " << ppi << std::endl;
	}
}


void createOpenXRSwapchain()
{
	// Try to get recommended resolution from OpenXR
	uint32_t viewCount = 2;
	xrEnumerateViewConfigurationViews(g_openXRState.instance, g_openXRState.systemId,
		XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, 0, &viewCount, nullptr);

	std::vector<XrViewConfigurationView> viewsConfig(viewCount, { XR_TYPE_VIEW_CONFIGURATION_VIEW });
	xrEnumerateViewConfigurationViews(g_openXRState.instance, g_openXRState.systemId,
		XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, viewCount, &viewCount, viewsConfig.data());

	uint32_t width = 0;
	uint32_t height = 0;

	// Meta Quest resolution  1832x1920 per eye, but steamvr uses higher values
	if (viewCount >= 2)
	{
		width = viewsConfig[0].recommendedImageRectWidth;
		height = viewsConfig[0].recommendedImageRectHeight;
		std::cout << "OpenXR recommended resolution: " << width << "x" << height << std::endl;

		// Force native Meta Quest 3 resolution (per eye)
		//width = 1832;
		//height = 1920;
		//std::cout << "Forcing Meta Quest 3 native resolution: " << width << "x" << height << std::endl;
	}
	else
	{
		width = 1832; //resolution from vr specs per eye
		height = 1920;
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

void setDefaultFrameInfo(FrameInfo& frameInfo,
	const glm::vec3& envRotation,
	const glm::vec4& clearColor,
	const glm::vec3& pointLightPos,
	bool pointLightEnabled = true,
	const glm::vec3& pointLightColor = glm::vec3(1.0f),
	float pointLightRadius = 0.0f)
{
	frameInfo.envRotation = envRotation;
	frameInfo.clearColor = clearColor;
	frameInfo.pointLightPos = glm::vec4(pointLightPos, pointLightRadius); // w = sphere radius
	frameInfo.pointLightColorEnabled = glm::vec4(
		pointLightEnabled ? pointLightColor : glm::vec3(0.0f),
		pointLightEnabled ? 1.0f : 0.0f
	);
}

void setPointLightState(FrameInfo& frameInfo, bool enabled)
{
	if (enabled)
	{
		frameInfo.pointLightColorEnabled.w = 1.0f; // Enable light
	}
	else
	{
		frameInfo.pointLightColorEnabled.w = 0.0f; // Disable light
	}
}

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
			int maxSamples{ 3 };
			int maxDepth{ 5 };
			bool showAxis{ false };
			glm::vec4 clearColor{ 1.F };
			//glm::vec3 envRotation{ -4.F, 35.5F, 121.F };
			glm::vec3 envRotation{ 0.F };
			bool denoiseApply{ true };
			bool denoiseFirstFrame{ true };
			int denoiseEveryNFrames{ 500 };
			int mode{ 0 }; // 0 = R-dominant, 1 = L-dominant, -1 = no reprojection
			bool pointLightEnabled{ true };
			glm::vec3 pointLightPos{ 5.4f, 2.1f, -0.5f };        // above scene by default
			glm::vec3 pointLightColor{ 300.0f, 300.0f, 300.0f }; 
			float pointLightRadius{ 0.5f }; // 0 = point light(hard shadow), >0 = sphere light
			int doDebug = 0; // 0 = none, 1 = show reprojection, 2 = show ray count heatmap
			bool eyeDominanceRight = true; // If true, right eye is dominant (primary), otherwise left eye is dominant
			bool enableReprojection = true;

			struct ExperimentState {
				bool active = false;
				int currentScene = 0;
				int currentSetup = 0;
				std::chrono::steady_clock::time_point currentCaseStart;
				std::chrono::steady_clock::time_point blackScreenStart;
				bool isBlackScreen = false;
				int blackScreenDurationMs = 1000;  // 1 second black screen
				int caseDurationMs = 15000;         // 15 seconds per case
				bool experimentComplete = false;
				bool firstCaseStarted = false;

				// For random ordering
				std::vector<int> shuffledSetups;    // Randomized order of setups
				int shuffledIndex = 0;              // Current position in shuffled order
			} experiment;

			int64_t timerMs = 0;
			std::chrono::steady_clock::time_point sessionStart;
			int sessionIndex = 0;
			bool sessionStarted{ false };        // true while the timer is running
			bool sessionFinished{ false };       // true after timer reached duration

			std::string logSessionName;
			std::filesystem::path logFolder;
			std::filesystem::path logFilePath;
			std::ofstream logFile;

			int sceneSetupIdx = 0;

			int sceneIdx = 0;
			// default camera values for the Sponza scene, will be overridden by UI
			glm::vec3 defaultEye{ 0.0f, 1.6f, 0.0f };
			glm::vec3 defaultCenter{ 10.0f, 1.6f, 0.0f };

		} m_settings;

;


		struct RayStatsGpu
		{
			uint32_t raysRPrimary{ 0 };
			uint32_t raysLPrimary{ 0 };
		} m_rayStatsCpu;

	public:
		OptixDenoiserEngine()
		{
			m_frameInfo.maxLuminance = 10.0F;
			//m_frameInfo.maxLuminance = 500.0F;
			m_frameInfo.clearColor = glm::vec4(1.F);
			m_frameInfo.envIntensity = 1.F;
			m_frameInfo.pointLightPos = glm::vec4(5.4f, 2.1f, -0.5f, 0.0f); // near a typical eye height
			m_frameInfo.pointLightColorEnabled = glm::vec4(glm::vec3(150.0f), 1.0f); // bright white, enabled
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

//#ifdef NVP_SUPPORTS_OPTIX7
#if defined(NVP_SUPPORTS_OPTIX9) || defined(NVP_SUPPORTS_OPTIX7)
			m_denoiser = std::make_unique<DenoiserOptix>();
			m_denoiser->setup(m_device, m_physicalDevice, m_app->getQueue(0).familyIndex);

			OptixDenoiserOptions d_options;
			d_options.guideAlbedo = 1u;
			d_options.guideNormal = 1u;
			/*m_denoiser->initOptiX(d_options, OPTIX_PIXEL_FORMAT_FLOAT4, true);
			m_denoiser->createSemaphore();
			m_denoiser->createCopyPipeline();*/
			if (!m_denoiser->initOptiX(d_options, OPTIX_PIXEL_FORMAT_FLOAT4, true)) {
				std::cerr << "OptiX denoiser failed to initialize — disabling denoiser." << std::endl;
				m_settings.denoiseApply = false;
			} else {
				m_denoiser->createSemaphore();
				m_denoiser->createCopyPipeline();
				}
#else
			m_settings.denoiseApply = false;
			LOGE("OptiX is not supported");
#endif // NVP_SUPPORTS_OPTIX7 || NVP_SUPPORTS_OPTIX9 

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
			if (g_enableXR && g_openXRState.swapchainWidth > 0)
			{
				//m_viewSize = glm::vec2(g_openXRState.swapchainWidth, g_openXRState.swapchainHeight);
				m_viewSize = glm::vec2(
					static_cast<float>(g_openXRState.swapchainWidth) * 2.0f,
					static_cast<float>(g_openXRState.swapchainHeight));
				std::cout << "GBuffer size set to: " << m_viewSize.x << "x" << m_viewSize.y << std::endl;
			}
			// Create resources
			createCommandBuffers();
			createGbuffers(m_viewSize);
			createVulkanBuffers();

			m_tonemapper->createComputePipeline();
			initialSetup();
		}

		void onDetach() override
		{
			vkDeviceWaitIdle(m_device);
			destroyResources();
		}

		void onResize(uint32_t width, uint32_t height) override
		{
			// In XR mode, onRenderVR manages GBuffer size (double-wide).
			// Don't let the window system override it.
			if (g_enableXR)
				return;

			// Skip if size hasn't actually changed
			if (m_gBuffers && m_gBuffers->getSize().width == width && m_gBuffers->getSize().height == height)
			{
				return;
			}
			std::cout << "Resizing for desktop: " << width << "x" << height << std::endl;
			createGbuffers({ width, height });

			m_tonemapper->updateComputeDescriptorSets(
				m_gBuffers->getDescriptorImageInfo(showDenoisedImage() ? eGbufDenoised : eGBufResult),
				m_gBuffers->getDescriptorImageInfo(eGBufLdr));

			writeRtxSet();
		}


		void onUIMenu() override
		{
			if (g_enableXR)
				return;

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
		void resetCamera()
		{
			CameraManip.setLookat(
				m_settings.defaultEye,
				m_settings.defaultCenter,
				glm::vec3(0.0f, 1.0f, 0.0f),
				true
			);
		}


		// Clear the "result" GBuffer (float4) to black so tonemapper outputs black.
		// Records commands into the provided command buffer.
		void clearResultToBlack(VkCommandBuffer cmd)
		{
			VkImage img = m_gBuffers->getColorImage(eGBufResult);

			// Transition from GENERAL (shader write/read) to TRANSFER_DST for clear
			VkImageMemoryBarrier toTransfer{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
			toTransfer.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
			toTransfer.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			toTransfer.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
			toTransfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			toTransfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			toTransfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			toTransfer.image = img;
			toTransfer.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

			vkCmdPipelineBarrier(cmd,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				0, 0, nullptr, 0, nullptr, 1, &toTransfer);

			VkClearColorValue clearColor{};
			clearColor.float32[0] = 0.0f;
			clearColor.float32[1] = 0.0f;
			clearColor.float32[2] = 0.0f;
			clearColor.float32[3] = 0.0f;

			VkImageSubresourceRange range{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
			vkCmdClearColorImage(cmd, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clearColor, 1, &range);

			// Transition back to GENERAL and make available for compute (tonemapper)
			VkImageMemoryBarrier toGeneral{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
			toGeneral.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			toGeneral.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
			toGeneral.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
			toGeneral.newLayout = VK_IMAGE_LAYOUT_GENERAL;
			toGeneral.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			toGeneral.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			toGeneral.image = img;
			toGeneral.subresourceRange = range;

			vkCmdPipelineBarrier(cmd,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
				0, 0, nullptr, 0, nullptr, 1, &toGeneral);
		}

		void fullExperiment() {
			// Initialize experiment state
			m_settings.experiment.active = true;
			m_settings.experiment.currentScene = 0;
			m_settings.experiment.currentSetup = 0;
			m_settings.experiment.isBlackScreen = false;
			m_settings.experiment.experimentComplete = false;
			m_settings.experiment.firstCaseStarted = true;

			m_settings.sessionStarted = true;
			m_settings.sessionFinished = false;
			m_settings.sessionStart = std::chrono::steady_clock::now();
			m_settings.timerMs = 0;

			// Create randomized setup order for first scene
			shuffleSetups();

			// Load first scene and first setup from shuffled order
			changeScene(m_settings.experiment.currentScene);
			m_settings.experiment.currentSetup = m_settings.experiment.shuffledSetups[0];
			m_settings.experiment.shuffledIndex = 0;
			sceneSetupHandler(m_settings.experiment.currentSetup);
			resetFrame();

			// Start timing for first case
			m_settings.experiment.currentCaseStart = std::chrono::steady_clock::now();

			// Recenter if XR
			if (g_enableXR)
			{
				recenterXRToIdentity();
			}

			// Log initial state
			logExperimentState();

			std::cout << "Experiment started: Scene " << m_settings.experiment.currentScene
				<< " (" << sceneNames[m_settings.experiment.currentScene] << ")"
				<< ", Setup " << m_settings.experiment.currentSetup << std::endl;
		}

		// Called every frame - this is your "loop" across frames
		void updateExperiment() {
			if (!m_settings.experiment.active || m_settings.experiment.experimentComplete)
				return;

			if (!m_settings.experiment.firstCaseStarted)
				return;

			auto now = std::chrono::steady_clock::now();

			if (m_settings.experiment.isBlackScreen) {
				// Check if black screen duration has elapsed
				auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
					now - m_settings.experiment.blackScreenStart).count();

				if (elapsed >= m_settings.experiment.blackScreenDurationMs) {
					// Black screen done - advance to next case
					m_settings.experiment.isBlackScreen = false;
					m_pushConst.showBlackScreen = 0;

					// THIS IS YOUR "LOOP" - advance to next iteration
					advanceToNextCase();
				}
			}
			else {
				// Check if case duration has elapsed
				auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
					now - m_settings.experiment.currentCaseStart).count();

				if (elapsed >= m_settings.experiment.caseDurationMs) {
					// Case done - start black screen
					m_settings.experiment.isBlackScreen = true;
					m_pushConst.showBlackScreen = 1;
					resetFrame();

					m_settings.experiment.blackScreenStart = now;
					std::cout << "Case complete - showing black screen..." << std::endl;
				}
			}

			// Update timer for UI
			if (!m_settings.experiment.isBlackScreen) {
				m_settings.timerMs = std::chrono::duration_cast<std::chrono::milliseconds>(
					now - m_settings.experiment.currentCaseStart).count();
			}
		}

		void shuffleSetups() {
			// Create array of setup indices
			m_settings.experiment.shuffledSetups = { 0, 1, 2, 3 };

			// Fisher-Yates shuffle
			std::random_device rd;
			std::mt19937 gen(rd());
			std::shuffle(m_settings.experiment.shuffledSetups.begin(),
				m_settings.experiment.shuffledSetups.end(),
				gen);

			m_settings.experiment.shuffledIndex = 0;

			// Log the order for reproducibility
			std::cout << "Setup order for scene " << m_settings.experiment.currentScene << ": ";
			for (int setup : m_settings.experiment.shuffledSetups) {
				std::cout << setup << " ";
			}
			std::cout << std::endl;
		}

		// This function mimics your nested loop iteration
		void advanceToNextCase() {
			// Move to next setup in shuffled order
			m_settings.experiment.shuffledIndex++;

			// Check if we've completed all setups for current scene
			if (m_settings.experiment.shuffledIndex >= 4) {
				// All setups done for this scene, move to next scene
				m_settings.experiment.currentScene++;

				// Check if all scenes are done
				if (m_settings.experiment.currentScene >= sceneCount) {
					// Experiment complete!
					m_settings.experiment.active = false;
					m_settings.experiment.experimentComplete = true;
					m_settings.sessionFinished = true;
					m_settings.sessionStarted = false;
					std::cout << "Experiment complete! All scenes and setups done." << std::endl;
					return;
				}

				// Load new scene
				std::cout << "Loading new scene: " << sceneNames[m_settings.experiment.currentScene] << std::endl;
				changeScene(m_settings.experiment.currentScene);

				// Create new randomized order for this scene
				shuffleSetups();
			}

			// Get next setup from shuffled order
			m_settings.experiment.currentSetup = m_settings.experiment.shuffledSetups[m_settings.experiment.shuffledIndex];

			// Apply the setup
			sceneSetupHandler(m_settings.experiment.currentSetup);
			resetFrame();

			// Start timing for this new case
			m_settings.experiment.currentCaseStart = std::chrono::steady_clock::now();

			// Log the new case
			logExperimentState();

			std::cout << "Started case: Scene " << m_settings.experiment.currentScene
				<< " (" << sceneNames[m_settings.experiment.currentScene] << ")"
				<< ", Setup " << m_settings.experiment.currentSetup
				<< " (shuffled index " << m_settings.experiment.shuffledIndex << "/3)" << std::endl;
		}


		void changeScene(int sceneIndex)
		{
			std::string scn_file = nvh::findFile(sceneFiles[sceneIndex], { ".", "..", "../..", "../../.." }, true);
			onFileDrop(scn_file.c_str());

			if (strcmp(sceneNames[sceneIndex], "Chess") == 0) {
				setPointLightState(m_frameInfo, false);
				m_settings.pointLightEnabled = false;

				// Canonical start camera for Chess (explicit values requested)
				glm::vec3 cameraPos(-3.005f, 1.455f, 0.15f);           // Eye
				glm::vec3 cameraCenter(14.31f, -2.9022f, 0.505);       // Center
				glm::vec3 up(0.0f, 1.0f, 0.0f);
				CameraManip.setLookat(cameraPos, cameraCenter, up, true);
			}
			else if (strcmp(sceneNames[sceneIndex], "Sponza") == 0) {
				setPointLightState(m_frameInfo, true);
				m_settings.pointLightEnabled = true;
				resetCamera();
			}
			else if (strcmp(sceneNames[sceneIndex], "City") == 0) {
				setPointLightState(m_frameInfo, true);
				m_settings.pointLightEnabled = false;
				resetCamera();
			}
			else {
				setPointLightState(m_frameInfo, true);
				m_settings.pointLightEnabled = true;
				resetCamera();

			}
			resetFrame();
		}

		void sceneSetupHandler(int idx = -1) {
			if (idx >= 0) {
				m_settings.sceneSetupIdx = idx;
			}
			else {
				// Randomize if no index provided
				m_settings.sceneSetupIdx = rand() % 4;
			}
			idx = rand() % 4 ? -1 : idx;
			switch (m_settings.sceneSetupIdx) {
			case 0:
				m_settings.enableReprojection = false;
				break;
			case 1:
				m_settings.enableReprojection = true;
				m_middleRadius = 30.0f;
				break;
			case 2:
				m_settings.enableReprojection = true;
				m_middleRadius = 45.0f;
				break;
			case 3:
				m_settings.enableReprojection = true;
				m_middleRadius = 60.0f;
				break;
			}

			setModeFromReproDom();
		}

		/*
			This is to send values correctly to the shader. Can be optimized, 
			but it will more time I want to spend now

		*/
		void setModeFromReproDom() {
			if (!m_settings.enableReprojection) {
				m_settings.mode = -1; // no reprojection	
			}
			else {
				if (m_settings.eyeDominanceRight) {
					m_settings.mode = 0; // R-dominant
				}
				else {
					m_settings.mode = 1; // L-dominant
				}
			}	
		}

		void setReproDomFromMode() {
			if (m_settings.mode == -1) {
				m_settings.enableReprojection = false;
			}
			else {
				m_settings.enableReprojection = true;
				if (m_settings.mode == 0) {
					m_settings.eyeDominanceRight = true;
				} 
				else if(m_settings.mode == 1) {
					m_settings.eyeDominanceRight = false;
				}
			}
		}



		void onUIRender() override
		{
			//if (g_enableXR) //	// In XR mode, the UI is rendered in-world, so skip the desktop UI.
			//{
			//	return;
			//}
			using namespace ImGuiH;

			bool reset{ false };
			// Pick under mouse cursor
			//if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left) || ImGui::IsKeyPressed(ImGuiKey_Space))
			//{
			//	screenPicking();
			//}
			/* ----------Buttons Pressed-----------*/
			{
				if (m_settings.sessionStarted && !m_settings.sessionFinished)
				{
					// Check for key presses and log them
					bool keyPressed = false;
					const char* keyName = nullptr;
					if (ImGui::IsKeyPressed(ImGuiKey_Space))
					{
						keyPressed = true;
						keyName = "space";
					}
					else if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow))
					{
						keyPressed = true;
						keyName = "left_arrow";
					}
					else if (ImGui::IsKeyPressed(ImGuiKey_RightArrow))
					{
						keyPressed = true;
						keyName = "right_arrow";
					}
					else if (ImGui::IsKeyPressed(ImGuiKey_UpArrow))
					{
						keyPressed = true;
						keyName = "up_arrow";
					}
					else if (ImGui::IsKeyPressed(ImGuiKey_DownArrow))
					{
						keyPressed = true;
						keyName = "down_arrow";
					}
					else if (ImGui::IsKeyPressed(ImGuiKey_Keypad1))
					{
						keyPressed = true;
						keyName = "numpad1";
					}
					else if (ImGui::IsKeyPressed(ImGuiKey_Keypad2))
					{
						keyPressed = true;
						keyName = "numpad2";
					}
					if (keyPressed && keyName)
					{
						appendSessionLogLine(keyName);
					}
				}
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
							{ return ImGui::SliderInt("#1", &m_settings.maxDepth, 1, 20); });
						reset |= PropertyEditor::entry("Samples", [&]
							{ return ImGui::SliderInt("#2", &m_settings.maxSamples, 1, 10); });
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
							{ return ImGui::ColorEdit3(
								"##Color", &m_settings.clearColor.x, ImGuiColorEditFlags_Float); },
							"Color multiplier");

						reset |= PropertyEditor::entry(
							"Rotation", [&]
							//{ return ImGui::SliderAngle("Rotation", &m_settings.envRotation); }, "Rotating the environment");
						{ 
								//return ImGui::SliderFloat("Rotation", &m_settings.envRotation, -10.0f, 10.0f); 
								return ImGui::SliderFloat3("##EnvRot", &m_settings.envRotation.x, -180.0f, 180.0f);
							}, "Rotating the environment");
						
						reset |= PropertyEditor::entry(
							"Light Intensity", [&]
							{ return ImGui::SliderFloat("Light Intensity", &m_frameInfo.envIntensity, 0.0f, 1000.0f); },
							"Multiplies HDR environment radiance (or point light power when point light is enabled)");

						reset |= PropertyEditor::entry(
							"Point Light Enabled", [&] { return ImGui::Checkbox("##ptEnable", &m_settings.pointLightEnabled); },
							"Toggle punctual emitter (in addition to HDR)");

						reset |= PropertyEditor::entry(
							"Point Light Pos", [&] { return ImGui::DragFloat3("##ptPos", &m_settings.pointLightPos.x, 0.1f); },
							"World position of the punctual emitter");

						reset |= PropertyEditor::entry(
							"Point Light Color (W)", [&] { return ImGui::ColorEdit3("##ptColor", &m_settings.pointLightColor.x, ImGuiColorEditFlags_HDR); },
							"Radiometric color/intensity for point light (linear)");
						reset |= PropertyEditor::entry(
							"Point Light Radius", [&] {
								return ImGui::DragFloat("##ptRadius", &m_settings.pointLightRadius,
									0.01f, 0.0f, 20.0f); },
									"Sphere radius for soft shadow penumbra (0 = hard shadow point light)");

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

					/*ImVec2 tumbnailSize = { 150 * m_gBuffers->getAspectRatio(), 150 };
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
					ImGui::Image(m_gBuffers->getDescriptorSet(eGbufDenoised), tumbnailSize);*/
				}
				if (ImGui::SliderFloat("Middle Radius", &m_middleRadius, 1.0f, 150.0f)) {
					reset = true;
				}
				ImGui::SliderInt("debug", &m_settings.doDebug, 0, 1);
				//if (ImGui::SliderFloat("Eye Separation (m)", &m_xrEyeSeparation, 0.05f, 0.075f, "%.3f")) {
				//	resetFrame(); // Flush accumulation to avoid ghosting
				//}
				if (ImGui::SliderInt("Scene", &m_settings.sceneIdx, 0, sceneCount - 1, sceneNames[m_settings.sceneIdx])) {
					changeScene(m_settings.sceneIdx);
				}
				if (ImGui::CollapsingHeader("Statistics", ImGuiTreeNodeFlags_DefaultOpen))
				{
					const float    fps = ImGui::GetIO().Framerate;
					// Prefer measured GPU counters when available, otherwise show estimates
					uint64_t displayR = m_hasMeasuredRays ? m_measuredRaysR : m_estRaysR;
					uint64_t displayL = m_hasMeasuredRays ? m_measuredRaysL : m_estRaysL;
					const double   totalPerSec = static_cast<double>(displayR + displayL) * fps;
					const char* domLabel = (m_settings.mode == 1) ? "Left  (dominant)" : "Right (dominant)";
					const char* nonDomLabel = (m_settings.mode == 1) ? "Right" : "Left ";

					ImGui::Text("%-18s %.2f M rays/frame", domLabel, static_cast<double>(displayR) * 1e-6);
					ImGui::Text("%-18s %.2f M rays/frame", nonDomLabel, static_cast<double>(displayL) * 1e-6);
					ImGui::Separator();
					ImGui::Text("Total/frame:       %.2f M rays", static_cast<double>(displayR + displayL) * 1e-6);
					ImGui::Text("Throughput:        %.2f M rays/sec", totalPerSec * 1e-6);
					ImGui::Text("                  (primary only, max estimate)");
				}
				static bool showDialog = false;
				if (ImGui::SliderInt("Modes", &m_settings.mode, -1, 2)) {
					setReproDomFromMode();
					reset = true; // request frame reset
				}
				if (ImGui::Checkbox("Enable Parallax Reprojection", &m_settings.enableReprojection))
				{
					setModeFromReproDom();
					reset = true; // request frame reset
				}

				// Eye dominance controls (disabled when reprojection is off)
					// Radio buttons for explicit eye dominance
				if (ImGui::RadioButton("Right eye dominant", m_settings.eyeDominanceRight))
				{
					m_settings.eyeDominanceRight = true;
					setModeFromReproDom();
					reset = true;
				}
				ImGui::SameLine();

				if (ImGui::RadioButton("Left eye dominant", !m_settings.eyeDominanceRight))
				{
					m_settings.eyeDominanceRight = false;
					setModeFromReproDom();
					reset = true;
				}
				
				ImGui::Separator();


				if (ImGui::CollapsingHeader("Setup", ImGuiTreeNodeFlags_DefaultOpen))
				{
					if (ImGui::Button("New Session"))
					{
						startNewSession();
					}

					ImGui::Separator();
					ImGui::Text("Session Preset:");
					if (ImGui::RadioButton("0 ", &m_settings.sceneSetupIdx, 0)) {
						sceneSetupHandler(m_settings.sceneSetupIdx);
					}
					ImGui::SameLine();
					if (ImGui::RadioButton("1 ", &m_settings.sceneSetupIdx, 1)) {
						sceneSetupHandler(m_settings.sceneSetupIdx);
					}
					ImGui::SameLine();
					if (ImGui::RadioButton("2 ", &m_settings.sceneSetupIdx, 2)) {
						sceneSetupHandler(m_settings.sceneSetupIdx);
					}
					ImGui::SameLine();
					if (ImGui::RadioButton("3 ", &m_settings.sceneSetupIdx, 3)) {
						sceneSetupHandler(m_settings.sceneSetupIdx);
					}
					if (!m_settings.experiment.active)
					{
						ImGui::SliderInt("Case duration", &m_settings.experiment.caseDurationMs, 2000, 30000);
					}

					ImGui::Text("Current session: %s", m_settings.logSessionName.c_str());
					ImGui::Text("Timer: %lld ms", static_cast<long long>(m_settings.timerMs));
				}

				if (ImGui::CollapsingHeader("Experiment Control", ImGuiTreeNodeFlags_DefaultOpen))
				{
					if (m_settings.experiment.active)
					{
						ImGui::Text("Experiment Running");
						ImGui::Text("Scene: %s (%d/%d)",
							sceneNames[m_settings.experiment.currentScene],
							m_settings.experiment.currentScene + 1, sceneCount);
						ImGui::Text("Setup: %d/4 (Random order: %d/4)",
							m_settings.experiment.currentSetup,
							m_settings.experiment.shuffledIndex + 1);

						// Show the randomized order for current scene
						ImGui::Text("Setup order: ");
						ImGui::SameLine();
						for (int i = 0; i < 4; i++) {
							if (i == m_settings.experiment.shuffledIndex) {
								ImGui::TextColored(ImVec4(0, 1, 0, 1), "[%d] ",
									m_settings.experiment.shuffledSetups[i]);
							}
							else {
								ImGui::SameLine();
								ImGui::Text("%d ", m_settings.experiment.shuffledSetups[i]);
							}
						}

						// ... rest of UI ...
					}
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

		static XrPosef mat4ToXrPosef(const glm::mat4& m)
		{
			glm::quat q = glm::quat_cast(m);
			XrPosef p{};
			p.orientation.x = q.x;
			p.orientation.y = q.y;
			p.orientation.z = q.z;
			p.orientation.w = q.w;
			p.position.x = m[3][0];
			p.position.y = m[3][1];
			p.position.z = m[3][2];
			return p;
		}

		// Recenter the reference space so current HMD pose becomes identity relative to the scene base.
// Call this after you set CameraManip to the desired start camera.
		void recenterXRToIdentity()
		{
			if (!m_scene->valid()) {
				std::cerr << "Cannot recenter: no scene loaded" << std::endl;
				return;
			}

			if (!g_enableXR || !g_openXRState.isInitialized() || g_openXRState.session == XR_NULL_HANDLE)
				return;

			// Wait a frame to get a current head pose
			XrFrameWaitInfo waitInfo{ XR_TYPE_FRAME_WAIT_INFO };
			XrFrameState frameState{ XR_TYPE_FRAME_STATE };
			XR_CHECK(xrWaitFrame(g_openXRState.session, &waitInfo, &frameState));

			XrFrameBeginInfo beginInfo{ XR_TYPE_FRAME_BEGIN_INFO };
			XR_CHECK(xrBeginFrame(g_openXRState.session, &beginInfo));

			// Locate views to read current head pose in the current reference space
			//XrView views[2] = { {XR_TYPE_VIEW}, {XR_TYPE_VIEW} };

			XrViewState viewState{ XR_TYPE_VIEW_STATE };
			uint32_t viewCountOutput = 0;
			XrViewLocateInfo viewLocate{ XR_TYPE_VIEW_LOCATE_INFO };
			viewLocate.viewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
			viewLocate.displayTime = frameState.predictedDisplayTime;
			viewLocate.space = g_openXRState.referenceSpace;
			XR_CHECK(xrLocateViews(g_openXRState.session, &viewLocate, &viewState, 2, &viewCountOutput, g_xrViews));

			const XrViewStateFlags validMask = XR_VIEW_STATE_ORIENTATION_VALID_BIT | XR_VIEW_STATE_POSITION_VALID_BIT;
			if (viewCountOutput < 1 || (viewState.viewStateFlags & validMask) != validMask)
			{
				XrFrameEndInfo endInfo{ XR_TYPE_FRAME_END_INFO };
				endInfo.displayTime = frameState.predictedDisplayTime;
				endInfo.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
				endInfo.layerCount = 0;
				endInfo.layers = nullptr;
				XR_CHECK(xrEndFrame(g_openXRState.session, &endInfo));
				return;
			}

			// headMat = current HMD pose (4x4) in reference space
			glm::mat4 headMat = xrPoseToMat4(g_xrViews[0].pose);

			// We want the head to become identity -> new reference pose = inverse(headMat)
			glm::mat4 invHead = glm::inverse(headMat);
			XrPosef newPose = mat4ToXrPosef(invHead);

			// Replace reference space with the offset so current head becomes origin
			if (g_openXRState.referenceSpace != XR_NULL_HANDLE)
			{
				xrDestroySpace(g_openXRState.referenceSpace);
				g_openXRState.referenceSpace = XR_NULL_HANDLE;
			}

			XrReferenceSpaceCreateInfo rsInfo{ XR_TYPE_REFERENCE_SPACE_CREATE_INFO };
			rsInfo.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_LOCAL;
			rsInfo.poseInReferenceSpace = newPose;
			XR_CHECK(xrCreateReferenceSpace(g_openXRState.session, &rsInfo, &g_openXRState.referenceSpace));

			// End the temporary frame
			XrFrameEndInfo endInfo{ XR_TYPE_FRAME_END_INFO };
			endInfo.displayTime = frameState.predictedDisplayTime;
			endInfo.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
			endInfo.layerCount = 0;
			endInfo.layers = nullptr;
			XR_CHECK(xrEndFrame(g_openXRState.session, &endInfo));
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

		float calculateFov(float leftDeg, float rightDeg, float _eyeSeparation, float nearZ) {
			// Calculate the horizontal FOV based on the left and right angles and the eye separation
			float leftTan = tanf(glm::radians(leftDeg));
			float rightTan = tanf(glm::radians(rightDeg));
			float fov = atan((rightTan - leftTan) / _eyeSeparation * nearZ) * 2.0f;
			return glm::degrees(fov);
		}

		void setHeadsetDisplayInfo(const HeadsetDisplayInfo& info)
		{
			g_headsetDisplayInfo.headsetName = info.headsetName;
			g_headsetDisplayInfo.displayWidth = info.displayWidth;
			g_headsetDisplayInfo.displayHeight = info.displayHeight;
			g_headsetDisplayInfo.pixelWidth = info.pixelWidth;
			g_headsetDisplayInfo.pixelHeight = info.pixelHeight;
			g_headsetDisplayInfo.ipd = info.ipd;
			g_headsetDisplayInfo.fovDegrees = info.fovDegrees;
			g_headsetDisplayInfo.fovRadians = info.fovRadians;
		}


		float getHeadsetViewerDistance() {
			if (!g_enableXR || !g_openXRState.isInitialized()) {
				return 0.5f; // Desktop monitor typical distance
			}

			// If we have physical panel dimensions, calculate accurately
			if (g_headsetDisplayInfo.displayWidth > 0.0f && g_headsetDisplayInfo.fovRadians > 0.0f) {
				// Using: distance = (panel_width / 2) / tan(fov / 2)
				float halfPanelWidth = g_headsetDisplayInfo.displayWidth * 0.5f;
				float halfFovRad = g_headsetDisplayInfo.fovRadians * 0.5f;
				float viewerDistance = halfPanelWidth / std::tan(halfFovRad);

				//std::cout << "Calculated viewer distance: " << viewerDistance
				//	<< "m (from panel width " << g_headsetDisplayInfo.displayWidth * 1000.0f
				//	<< "mm and FOV " << g_headsetDisplayInfo.fovDegrees << "°)" << std::endl;
				return viewerDistance;
			}

			// Fallback for Quest 3S family: known fixed focus distance
			const std::string hn = g_headsetDisplayInfo.headsetName;
			if (hn.find("Quest") != std::string::npos) {
				return 1.3f; // Meta Quest family fixed focus distance
			}

			// Generic VR headset fallback
			return 1.5f;
		}

		void setStereoViews(XrView views[]) {
			const auto& bounds = m_scene->getSceneBounds();
			float sceneRadius = bounds.radius();

			//const float nearZ = 0.1f;
			//const float farZ = 1000.0f;

			// Dynamic near/far based on scene size
			const float nearZ = sceneRadius * 0.001f;  // 0.1% of scene radius
			const float farZ = sceneRadius * 100.0f;   // 100x scene radius

			if (!m_xrProjCached) {
				m_cachedLeftProj = xrFovToProjMatrix(views[0].fov, nearZ, farZ);
				m_cachedLeftProj[1][1] *= -1;
				m_cachedRightProj = xrFovToProjMatrix(views[1].fov, nearZ, farZ);
				m_cachedRightProj[1][1] *= -1;
				m_cachedLeftProjInv = glm::inverse(m_cachedLeftProj);
				m_cachedRightProjInv = glm::inverse(m_cachedRightProj);
				m_xrProjCached = true;
			}

			// Calculate REAL horizontal FOV from OpenXR (asymmetric)
			float leftFovH = glm::degrees(views[0].fov.angleRight - views[0].fov.angleLeft);
			float rightFovH = glm::degrees(views[1].fov.angleRight - views[1].fov.angleLeft);
			g_headsetDisplayInfo.fovDegrees = (leftFovH + rightFovH) * 0.5f;
			g_headsetDisplayInfo.fovRadians = glm::radians(g_headsetDisplayInfo.fovDegrees);

			CameraManip.setFov(leftFovH);


			// Get the desktop camera transform as our scene base position
			glm::vec3 sceneEye = CameraManip.getEye(); // Position of the Camera
			glm::vec3 sceneCenter = CameraManip.getCenter(); // Point the camera is looking at
			glm::vec3 sceneUp = CameraManip.getUp(); // Up direction for the camera
			// Build a scene-space transform: position at sceneEye, looking toward sceneCenter
			glm::mat4 sceneBaseMat = glm::inverse(glm::lookAt(sceneEye, sceneCenter, sceneUp));

			// Left eye: combine scene base with XR head pose and avoid double inverse: viewInv IS leftWorld
			glm::mat4 leftWorld = sceneBaseMat * xrPoseToMat4(views[0].pose);
			glm::vec2 clip = CameraManip.getClipPlanes();
			m_frameInfo.clipNear = clip.x;
			m_frameInfo.clipFar = clip.y;
			m_frameInfo.view = glm::inverse(leftWorld);
			m_frameInfo.viewInv = leftWorld;                 // ✅ No extra inverse
			m_frameInfo.proj = m_cachedLeftProj;          // ✅ Cached
			m_frameInfo.projInv = m_cachedLeftProjInv;       // ✅ Cached
			m_frameInfo.camPos = glm::vec4(leftWorld[3]);

			// Right eye — same pattern
			glm::mat4 rightWorld = sceneBaseMat * xrPoseToMat4(views[1].pose);
			m_frameInfo.view2 = glm::inverse(rightWorld);
			m_frameInfo.view2Inv = rightWorld;               // ✅ No extra inverse
			m_frameInfo.proj2 = m_cachedRightProj;        // ✅ Cached
			m_frameInfo.proj2Inv = m_cachedRightProjInv;     // ✅ Cached
			m_frameInfo.camPos2 = glm::vec4(rightWorld[3]);

			setDefaultFrameInfo(m_frameInfo,
				m_settings.envRotation,
				m_settings.clearColor,
				m_settings.pointLightPos,
				m_settings.pointLightEnabled,
				m_settings.pointLightColor,
				m_settings.pointLightRadius
			);

			updateRayCounters();  // must be after setDefaultFrameInfo, before vkCmdUpdateBuffer

			// Calculate REAL IPD from actual eye positions
			glm::vec3 leftEyePos(views[0].pose.position.x, views[0].pose.position.y, views[0].pose.position.z);
			glm::vec3 rightEyePos(views[1].pose.position.x, views[1].pose.position.y, views[1].pose.position.z);
			g_headsetDisplayInfo.ipd = glm::distance(leftEyePos, rightEyePos);
			//m_xrEyeSeparation = g_headsetDisplayInfo.ipd;

			const VkExtent2D frameSize = m_gBuffers->getSize();
			presetConstShaderValues(
				m_frameInfo, 
				g_headsetDisplayInfo.fovDegrees, 
				g_headsetDisplayInfo.ipd, 
				m_middleRadius, 
				frameSize
			);

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
			// Left half = left eye, Right half = right eye
			//uint32_t requiredWidth = g_openXRState.swapchainWidth;
			uint32_t requiredWidth = g_openXRState.swapchainWidth * 2;
			uint32_t requiredHeight = g_openXRState.swapchainHeight;
			if (m_gBuffers->getSize().width != requiredWidth ||
				m_gBuffers->getSize().height != requiredHeight) {
				vkEndCommandBuffer(vkCmd);
				createGbuffers({ requiredWidth, requiredHeight });

				m_tonemapper->updateComputeDescriptorSets(
					m_gBuffers->getDescriptorImageInfo(showDenoisedImage() ? eGbufDenoised : eGBufResult),
					m_gBuffers->getDescriptorImageInfo(eGBufLdr));
				writeRtxSet();

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
			//XrView views[2] = { {XR_TYPE_VIEW}, {XR_TYPE_VIEW} };
			//g_xrViews[0] = views[0];
			//g_xrViews[1] = views[1];
			uint32_t viewCountOutput = 0;

			XrViewLocateInfo viewLocateInfo{ XR_TYPE_VIEW_LOCATE_INFO };
			viewLocateInfo.viewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
			viewLocateInfo.displayTime = xrFrameState.predictedDisplayTime;
			viewLocateInfo.space = g_openXRState.referenceSpace;

			XrViewState xrViewState{ XR_TYPE_VIEW_STATE };
			XR_CHECK(xrLocateViews(g_openXRState.session, &viewLocateInfo, &xrViewState, 2, &viewCountOutput, g_xrViews));

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
			{
				setStereoViews(g_xrViews);
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

			//float tanLeft = tanf(views[0].fov.angleLeft);   // negative
			//float tanRight = tanf(views[0].fov.angleRight);  // positive
			//float horizontalFovRad = atanf(tanRight) - atanf(tanLeft); // total horizontal span

			m_pushConst.maxDepth = m_settings.maxDepth;
			m_pushConst.maxSamples = m_settings.maxSamples;
			m_pushConst.frame = m_frame;
			m_pushConst.middleRadius = m_middleRadius;
			m_pushConst.mode = m_settings.mode;
			m_pushConst.doDebug = m_settings.doDebug;

			vkCmdFillBuffer(vkCmd, m_bRayStats.buffer, 0, sizeof(RayStatsGpu), 0);
			VkBufferMemoryBarrier statsResetBarrier{ VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER };
			statsResetBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			statsResetBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
			statsResetBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			statsResetBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			statsResetBarrier.buffer = m_bRayStats.buffer;
			statsResetBarrier.offset = 0;
			statsResetBarrier.size = sizeof(RayStatsGpu);

			vkCmdPipelineBarrier(vkCmd,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
				0, 0, nullptr, 1, &statsResetBarrier, 0, nullptr);

			raytraceScene(vkCmd);

			VkBufferMemoryBarrier statsWriteBarrier{ VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER };
			statsWriteBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
			statsWriteBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
			statsWriteBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			statsWriteBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			statsWriteBarrier.buffer = m_bRayStats.buffer;
			statsWriteBarrier.offset = 0;
			statsWriteBarrier.size = sizeof(RayStatsGpu);

			vkCmdPipelineBarrier(vkCmd,
				VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				0, 0, nullptr, 1, &statsWriteBarrier, 0, nullptr);

			VkBufferCopy copyRegion{};
			copyRegion.srcOffset = 0;
			copyRegion.dstOffset = 0;
			copyRegion.size = sizeof(RayStatsGpu);
			vkCmdCopyBuffer(vkCmd, m_bRayStats.buffer, m_bRayStatsReadback.buffer, 1, &copyRegion);


#if defined(NVP_SUPPORTS_OPTIX9) || defined(NVP_SUPPORTS_OPTIX7)
			// Submit cmd[0]: signal CUDA when raytracing is done
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
				//VkSemaphore tlSemaphore = m_denoiser->getTLSemaphore();
				//VkSemaphoreWaitInfo waitInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO };
				//waitInfo.semaphoreCount = 1;
				//waitInfo.pSemaphores = &tlSemaphore;
				//waitInfo.pValues = &m_fenceValue;
				//vkWaitSemaphores(m_device, &waitInfo, UINT64_MAX);

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

			// Blit LDR → XR swapchain left half of LDR to left eye, right half to right eye
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

			{	// Final layout transitions
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

			{	// Restore LDR to GENERAL for next frame
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
			vkEndCommandBuffer(vkCmd); // End command buffer BEFORE releasing swapchain

			// Submit GPU work and wait
			if (m_vrFence == VK_NULL_HANDLE)
			{
				VkFenceCreateInfo fci{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
				vkCreateFence(m_device, &fci, nullptr, &m_vrFence);
			} else {
				vkResetFences(m_device, 1, &m_vrFence);
			}

#if defined(NVP_SUPPORTS_OPTIX9) || defined(NVP_SUPPORTS_OPTIX7)
			if (m_settings.denoiseApply) {
				// ✅ GPU waits for CUDA semaphore — no CPU stall!
				VkSemaphoreSubmitInfo wait_sem{
					.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO_KHR,
					.semaphore = m_denoiser->getTLSemaphore(),
					.value = m_fenceValue,
					.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT_KHR,
				};
				VkCommandBufferSubmitInfo cmd1_info{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO_KHR, nullptr, vkCmd };
				VkSubmitInfo2KHR submit1{
					.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2_KHR,
					.waitSemaphoreInfoCount = 1,
					.pWaitSemaphoreInfos = &wait_sem,
					.commandBufferInfoCount = 1,
					.pCommandBufferInfos = &cmd1_info,
				};
				vkQueueSubmit2(m_app->getQueue(0).queue, 1, &submit1, m_vrFence);
			} else
#endif
			{
				VkCommandBufferSubmitInfo cmdInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
				cmdInfo.commandBuffer = vkCmd;
				VkSubmitInfo2 submit2{ VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
				submit2.commandBufferInfoCount = 1;
				submit2.pCommandBufferInfos = &cmdInfo;
				vkQueueSubmit2(m_app->getQueue(0).queue, 1, &submit2, m_vrFence);
			}
				vkWaitForFences(m_device, 1, &m_vrFence, VK_TRUE, UINT64_MAX);

				if (m_bRayStatsReadback.memHandle)
				{
					void* mapped = m_alloc->getMemoryAllocator()->map(m_bRayStatsReadback.memHandle);
					if (mapped)
					{
						const RayStatsGpu* rs = reinterpret_cast<const RayStatsGpu*>(mapped);
						m_rayStatsCpu = *rs;

						// Save measured counters separately and mark as available for UI
						m_measuredRaysR = static_cast<uint64_t>(m_rayStatsCpu.raysRPrimary);
						m_measuredRaysL = static_cast<uint64_t>(m_rayStatsCpu.raysLPrimary);
						m_hasMeasuredRays = true;

						m_alloc->getMemoryAllocator()->unmap(m_bRayStatsReadback.memHandle);
					}
				}
			
			{	// Release swapchain AFTER GPU finishes
				XrSwapchainImageReleaseInfo releaseInfo{ XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO };
				XR_CHECK(xrReleaseSwapchainImage(g_openXRState.swapchain, &releaseInfo));
			}

			// Build projection layer views
			projectionViews.resize(2);
			for (uint32_t eye = 0; eye < 2; ++eye)
			{
				XrCompositionLayerProjectionView pv{ XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW };
				pv.next = nullptr;
				pv.pose = g_xrViews[eye].pose;
				pv.fov = g_xrViews[eye].fov;

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

			if (m_bRayStatsReadback.memHandle)
			{
				void* mapped = m_alloc->getMemoryAllocator()->map(m_bRayStatsReadback.memHandle);
				if (mapped)
				{
					const RayStatsGpu* rs = reinterpret_cast<const RayStatsGpu*>(mapped);
					m_rayStatsCpu = *rs;

					// Save measured counters separately for UI
					m_measuredRaysR = static_cast<uint64_t>(m_rayStatsCpu.raysRPrimary);
					m_measuredRaysL = static_cast<uint64_t>(m_rayStatsCpu.raysLPrimary);
					m_hasMeasuredRays = true;

					m_alloc->getMemoryAllocator()->unmap(m_bRayStatsReadback.memHandle);
				}
			}

			// Using local command buffer for the frame
			const CommandFrame& commandFrame = m_commandFrames[m_app->getFrameCycleIndex()];
			VkCommandBuffer cmd = commandFrame.cmdBuffer[0];
			vkResetCommandPool(m_device, commandFrame.cmdPool, 0);
			VkCommandBufferBeginInfo begin_info{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, 0, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
			vkBeginCommandBuffer(cmd, &begin_info);

			auto scope_dbg = m_dutil->DBG_SCOPE(cmd);

			// Get camera info
			float view_aspect_ratio = (m_viewSize.x * 0.5f) / m_viewSize.y;
			//float eyeOffset = 0.032f; // Half IPD in meters (~64mm total)
			const float desktopEyeSeparation = g_headsetDisplayInfo.ipd;
			float eyeOffset = desktopEyeSeparation * 0.5f; // Half IPD for stereo offset
			const VkExtent2D frameSize = m_gBuffers->getSize();

			glm::vec3 eyeMid = CameraManip.getEye();
			glm::vec3 center = CameraManip.getCenter();
			glm::vec3 up = CameraManip.getUp();

			// Compute camera's local right vector (parallel to ground plane)
			glm::vec3 forward = glm::normalize(center - eyeMid);
			glm::vec3 right = glm::normalize(glm::cross(forward, up));

			CameraManip.setFov(104);


			// Parallel stereo: offset both eye AND center by the same amount
			// This keeps both cameras looking in the same direction (no toe-in)
			glm::vec3 eyeLeft = eyeMid - right * eyeOffset;
			glm::vec3 centerLeft = center - right * eyeOffset;

			glm::vec3 eyeRight = eyeMid + right * eyeOffset;
			glm::vec3 centerRight = center + right * eyeOffset;

			// Left eye
			glm::vec2 clip = CameraManip.getClipPlanes();

			m_frameInfo.clipNear = clip.x; 
			m_frameInfo.clipFar = clip.y;
			m_frameInfo.view = glm::lookAt(eyeLeft, centerLeft, up);
			m_frameInfo.viewInv = glm::inverse(m_frameInfo.view);
			m_frameInfo.proj = glm::perspectiveRH_ZO(glm::radians(CameraManip.getFov()), view_aspect_ratio, clip.x, clip.y);
			m_frameInfo.proj[1][1] *= -1;
			m_frameInfo.projInv = glm::inverse(m_frameInfo.proj);
			m_frameInfo.camPos = glm::vec4(eyeLeft, 0.0f);

			// Right eye
			m_frameInfo.view2 = glm::lookAt(eyeRight, centerRight, up);
			m_frameInfo.view2Inv = glm::inverse(m_frameInfo.view2);
			m_frameInfo.proj2 = glm::perspectiveRH_ZO(glm::radians(CameraManip.getFov()), view_aspect_ratio, clip.x, clip.y);
			m_frameInfo.proj2[1][1] *= -1;
			m_frameInfo.proj2Inv = glm::inverse(m_frameInfo.proj2);
			m_frameInfo.camPos2 = glm::vec4(eyeRight, 0.0f);

			presetConstShaderValues(m_frameInfo, CameraManip.getFov(), desktopEyeSeparation, m_middleRadius, frameSize);

			setDefaultFrameInfo(m_frameInfo,
				m_settings.envRotation,
				m_settings.clearColor,
				m_settings.pointLightPos,
				m_settings.pointLightEnabled,
				m_settings.pointLightColor,
				m_settings.pointLightRadius);

			updateRayCounters();  // must be after setDefaultFrameInfo, before vkCmdUpdateBuffer

			
			vkCmdUpdateBuffer(cmd, m_bFrameInfo.buffer, 0, sizeof(FrameInfo), &m_frameInfo);

			m_pushConst.maxDepth = m_settings.maxDepth;
			m_pushConst.maxSamples = m_settings.maxSamples;
			m_pushConst.frame = m_frame;
			m_pushConst.middleRadius = m_middleRadius;
			m_pushConst.mode = m_settings.mode;
			m_pushConst.doDebug = m_settings.doDebug;

			vkCmdFillBuffer(cmd, m_bRayStats.buffer, 0, sizeof(RayStatsGpu), 0);

			VkBufferMemoryBarrier statsResetBarrier{ VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER };
			statsResetBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			statsResetBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
			statsResetBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			statsResetBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			statsResetBarrier.buffer = m_bRayStats.buffer;
			statsResetBarrier.offset = 0;
			statsResetBarrier.size = sizeof(RayStatsGpu);

			vkCmdPipelineBarrier(cmd,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
				0, 0, nullptr, 1, &statsResetBarrier, 0, nullptr);

			raytraceScene(cmd);

			VkBufferMemoryBarrier statsWriteBarrier{ VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER };
			statsWriteBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
			statsWriteBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
			statsWriteBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			statsWriteBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			statsWriteBarrier.buffer = m_bRayStats.buffer;
			statsWriteBarrier.offset = 0;
			statsWriteBarrier.size = sizeof(RayStatsGpu);

			vkCmdPipelineBarrier(cmd,
				VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				0, 0, nullptr, 1, &statsWriteBarrier, 0, nullptr);

			VkBufferCopy copyRegion{};
			copyRegion.srcOffset = 0;
			copyRegion.dstOffset = 0;
			copyRegion.size = sizeof(RayStatsGpu);
			vkCmdCopyBuffer(cmd, m_bRayStats.buffer, m_bRayStatsReadback.buffer, 1, &copyRegion);

#if defined(NVP_SUPPORTS_OPTIX9) || defined(NVP_SUPPORTS_OPTIX7)
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
#endif // OPTIX7 and OPTIX9

			// Apply tonemapper - take GBuffer-X and output to GBuffer-0
			m_tonemapper->runCompute(cmd, m_gBuffers->getSize());

			// End of the first or second command buffer
			vkEndCommandBuffer(cmd);
			VkCommandBufferSubmitInfo submit_info{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO_KHR };
			submit_info.commandBuffer = cmd;
			m_app->prependCommandBuffer(submit_info); // Prepend to the frame command buffer
		}

		/*
			Set values that will be sent to the shader as push constants
			and that wont change between frames, such as the number of samples or the max ray depth.
		*/

		void presetConstShaderValues(FrameInfo& fi, float fovDegrees, float eyeSeparation,
			float middleRadiusPct, VkExtent2D size) {
			fi.fovDegrees = fovDegrees;
			fi.fovRadians = glm::radians(fovDegrees);
			fi.eyeSeparation = eyeSeparation;

			const float width = static_cast<float>(size.width);
			const float height = static_cast<float>(size.height);
			const float halfWidth = width * 0.5f;  // per-eye width in your double-wide layout

			fi.halfWidthPixels = halfWidth;
			fi.invClipRange = (fi.clipFar > fi.clipNear) ? (1.0f / (fi.clipFar - fi.clipNear)) : 0.0f;

			float tanHalf = tanf(fi.fovRadians * 0.5f);
			if (tanHalf < 1e-4f)
				tanHalf = 1e-4f;
			fi.tanHalfFov = tanHalf;

			fi.focalLengthPixels = (halfWidth * 0.5f) / fi.tanHalfFov;

			// Get dynamic viewer distance instead of hardcoded HOST_VIEWER_DISTANCE
			const float viewerDistance = getHeadsetViewerDistance();

			/* Calculate Total screen distance in pixels
				viewer_distance = (screen_width_per_eye / 2) / tan(FOV/2)
				pixels_per_meter = (screen_width_per_eye / 2) / (viewer_distance * tan(FOV/2))
			*/
			const float pixelsPerMeter = (halfWidth * 0.5f) / (viewerDistance * fi.tanHalfFov);
			fi.screenDistancePixels = viewerDistance * pixelsPerMeter;

			const float maxAngleRad = glm::radians(HOST_MAX_COMFORTABLE_PARALLAX_ANGLE);
			fi.maxComfortableParallaxPixels = 2.0f * fi.screenDistancePixels * tanf(maxAngleRad * 0.5f);

			// Convert UI value to circle diameter in degrees: 10 units = 1 degree
			// The radius is half the diameter TODO maybe fix name, radius might be confusing since it's actually diameter in degrees, but it will be converted to radius in pixels later
			const float radiusInDegrees = middleRadiusPct / 2.0f;

			// Convert degrees to pixels
			const float pixelRange = (halfWidth < height) ? halfWidth : height;
			const float pixelsPerDegree = pixelRange / fovDegrees;
			fi.reprojectionRadiusPixels = radiusInDegrees * pixelsPerDegree;

			// Store the viewer distance in frameInfo for debugging/UI
			// (you may need to add this field to FrameInfo struct)
			// fi.viewerDistance = viewerDistance;
		}


		void onRender(VkCommandBuffer cmd) override
		{
			updateExperiment();
			if (g_enableXR)
			{
				onRenderVR(cmd);
			}
			else
			{
				onRenderNonVR(cmd);
			}
		}

	private:
		static int getLastSessionId(const std::filesystem::path& root = std::filesystem::current_path())
		{
			int lastId = 0;

			for (const auto& entry : std::filesystem::directory_iterator(root))
			{
				if (!entry.is_directory())
					continue;

				const std::string name = entry.path().filename().string();
				if (name.rfind("log-", 0) != 0)
					continue;

				// Expected: log-0001-20240612-153000
				const size_t secondDash = name.find('-', 4);
				if (secondDash == std::string::npos)
					continue;

				const std::string idPart = name.substr(4, secondDash - 4);

				try
				{
					lastId = std::max(lastId, std::stoi(idPart));
				}
				catch (...)
				{
					// Ignore malformed folders
				}
			}

			return lastId;
		}


		void updateSessionTimer()
		{
			if (!m_settings.sessionStarted || m_settings.sessionFinished)
				return;

			m_settings.timerMs = std::chrono::duration_cast<std::chrono::milliseconds>(
				std::chrono::steady_clock::now() - m_settings.sessionStart).count();

			if (m_settings.timerMs >= m_settings.experiment.caseDurationMs)
			{
				// End session
				m_settings.sessionFinished = true;
				m_settings.sessionStarted = false;
				appendSessionLogLine("session_end");
				std::cout << "Session finished (timer reached)\n";
			}
		}

		void initialSetup() {
			changeScene(0); // Load the first scene for initial setup
			resetFrame();
		}



		/*
		 Create a .txt file in a folder with same name log-id-timestamp, where the id is ordered by timestamp.
		 For example:
		 log-0001-20240612-153000/log-0001-20240612-153000.txt
		 The file will contain the log of the benchmark, adding a line everytime a key is pressed
		 The line will contain the following information:
			timer - key pressed - mode - scene name
		 for example:
			9364 - right_arrow - No Reprojection - Chess
		*/
		void startNewSession()
		{
			// Stop any running session
			m_settings.sessionStarted = false;
			m_settings.sessionFinished = false;
			m_settings.timerMs = 0;

			resetFrame();

			// Setup logging
			const int lastId = getLastSessionId();
			m_settings.sessionIndex = lastId + 1;

			auto now = std::chrono::system_clock::now();
			std::time_t tt = std::chrono::system_clock::to_time_t(now);
			std::tm tm{};
#ifdef _WIN32
			localtime_s(&tm, &tt);
#else
			localtime_r(&tt, &tm);
#endif

			std::ostringstream name;
			name << "log-"
				<< std::setw(4) << std::setfill('0') << m_settings.sessionIndex
				<< "-"
				<< std::put_time(&tm, "%Y%m%d-%H%M%S");

			m_settings.logSessionName = name.str();
			m_settings.logFolder = m_settings.logSessionName;
			m_settings.logFilePath = m_settings.logFolder / (m_settings.logSessionName + ".txt");

			if (m_settings.logFile.is_open())
			{
				m_settings.logFile.close();
			}

			std::filesystem::create_directories(m_settings.logFolder);
			m_settings.logFile.open(m_settings.logFilePath, std::ios::out | std::ios::trunc);
			if (m_settings.logFile.is_open())
			{
				m_settings.logFile << "TimerMs, Key Pressed, Reprojection (eye dominance), Circle Degree, Scene Name\n";
				m_settings.logFile.flush();
			}

			std::cout << "New session prepared: " << m_settings.logSessionName << std::endl;

			// THIS IS THE KEY CHANGE: Just call fullExperiment() and it handles everything
			fullExperiment();
		}

		void logExperimentState()
		{
			if (!m_settings.logFile.is_open())
				return;

			const char* modeText = "Unknown";
			switch (m_settings.mode)
			{
			case 1:  modeText = "Left Dominant"; break;
			case 0:  modeText = "Right Dominant"; break;
			case -1: modeText = "No Reprojection"; break;
			}

			const char* circleDegree = "Unknown";
			switch (m_settings.sceneSetupIdx) {
			case 0: circleDegree = "Not Used"; break;
			case 1: circleDegree = "30 degrees Circle"; break;
			case 2: circleDegree = "45 degrees Circle"; break;
			case 3: circleDegree = "60 degrees Circle"; break;
			}

			const char* sceneName = sceneNames[m_settings.experiment.currentScene];

			m_settings.logFile
				<< "0" << ","  // Timer starts at 0 for each case
				<< sceneName << ","
				<< m_settings.experiment.currentSetup << ","
				<< modeText << ","
				<< circleDegree << ","
				<< "ShuffledIndex:" << m_settings.experiment.shuffledIndex
				<< "\n";
			m_settings.logFile.flush();
		}

		void appendSessionLogLine(const char* keyPressed)
		{
			if (!m_settings.logFile.is_open())
				return;

			// Update timer to current time
			auto now = std::chrono::steady_clock::now();
			m_settings.timerMs = std::chrono::duration_cast<std::chrono::milliseconds>(
				now - m_settings.sessionStart).count();

			const char* modeText = "Unknown";
			switch (m_settings.mode)
			{
			case 1:  modeText = "Left Dominant"; break;
			case 0:  modeText = "Right Dominant"; break;
			case -1: modeText = "No Reprojection"; break;
			}

			const char* circleDegree = "Unknown";
			switch (m_settings.sceneSetupIdx) {
			case 0: circleDegree = "No circle"; break;
			case 1: circleDegree = "30deg circle"; break;
			case 2: circleDegree = "45deg circle"; break;
			case 3: circleDegree = "60deg circle"; break;
			}

			const char* sceneName = "Unknown";
			if (m_settings.sceneIdx >= 0 && m_settings.sceneIdx < sceneCount)
				sceneName = sceneNames[m_settings.sceneIdx];

			// Write the log line
			m_settings.logFile
				<< m_settings.timerMs << " - "
				<< keyPressed << " - "
				<< modeText << " - "
				<< circleDegree << " - "
				<< sceneName << "\n";

			m_settings.logFile.flush();

			std::cout << "Logged: " << m_settings.timerMs << "ms - " << keyPressed
				<< " - " << modeText << " - " << circleDegree << " - " << sceneName << std::endl;
		}

		void createScene(const std::string& filename, float sceneScale = 1.0f)
		{
			m_scene->load(filename);

			// ------- Apply a uniform scene scale here -------
			// Change this value to scale the whole glTF scene (e.g. 0.5 = half size, 2.0 = double size).
			// if scene is beautifulGame, use scale 3.0f, otherwise, 1.0f
			if (filename.find("Untitled") != std::string::npos) {
				sceneScale = 7.0f;
			}

			// Get the root node, update its scale, and set it back.
			tinygltf::Node rootNode = m_scene->getSceneRootNode();
			if (rootNode.scale.size() == 3)
			{
				rootNode.scale[0] *= static_cast<double>(sceneScale);
				rootNode.scale[1] *= static_cast<double>(sceneScale);
				rootNode.scale[2] *= static_cast<double>(sceneScale);
			}
			else
			{
				rootNode.scale = { static_cast<double>(sceneScale),
								   static_cast<double>(sceneScale),
								   static_cast<double>(sceneScale) };
			}
			m_scene->setSceneRootNode(rootNode);
			// ------- end scale tweak -------

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
			//std::vector<VkFormat> color_buffers = {
			//	VK_FORMAT_B8G8R8A8_UNORM,      // LDR
			//	VK_FORMAT_R32G32B32A32_SFLOAT, // Result
			//	VK_FORMAT_R16G16B16A16_SFLOAT,  // Albedo
			//	VK_FORMAT_R16G16B16A16_SFLOAT,  // Normal
			//	VK_FORMAT_R16G16B16A16_SFLOAT,  // Depth
			//	VK_FORMAT_R16G16B16A16_SFLOAT,  // Disparity
			//	VK_FORMAT_R16G16B16A16_SFLOAT,  // Denoised
			//};
			std::vector<VkFormat> color_buffers = {
				VK_FORMAT_B8G8R8A8_UNORM,       // LDR
				VK_FORMAT_R32G32B32A32_SFLOAT,  // Result (denoiser needs FLOAT4)
				VK_FORMAT_R16G16B16A16_SFLOAT,  // Albedo
				VK_FORMAT_R16G16B16A16_SFLOAT,  // Normal
				VK_FORMAT_R32_SFLOAT,           // Depth — single channel is sufficient
				VK_FORMAT_R8G8B8A8_UNORM,       // Disparity — debug visualization only
				VK_FORMAT_R16G16B16A16_SFLOAT,  // Denoised
			};

			// Creation of the GBuffers
			m_gBuffers = std::make_unique<nvvkhl::GBuffer>(m_device, m_alloc.get(), vk_size, color_buffers, depth_format);

#if defined(NVP_SUPPORTS_OPTIX9) || defined(NVP_SUPPORTS_OPTIX7)
			// Only allocate denoiser buffers when denoising is actually enabled
			//if (m_settings.denoiseApply && !g_enableXR)
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

			m_bRayStats = m_alloc->createBuffer(
				sizeof(RayStatsGpu),
				VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

			m_bRayStatsReadback = m_alloc->createBuffer(
				sizeof(RayStatsGpu),
				VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

			m_dutil->DBG_NAME(m_bFrameInfo.buffer);
			m_dutil->DBG_NAME(m_bRayStats.buffer);
			m_dutil->DBG_NAME(m_bRayStatsReadback.buffer);
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
			d->addBinding(RtxBindings::eRayStats, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);

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
			VkDescriptorBufferInfo rtxStats_info{ m_bRayStats.buffer, 0, VK_WHOLE_SIZE };

			std::vector<VkWriteDescriptorSet> writes;
			writes.emplace_back(d->makeWrite(0, RtxBindings::eTlas, &desc_as_info));
			writes.emplace_back(d->makeWrite(0, RtxBindings::eOutImage, &image_info));
			// #OPTIX_D
			writes.emplace_back(d->makeWrite(0, RtxBindings::eOutAlbedo, &albedo_info));
			writes.emplace_back(d->makeWrite(0, RtxBindings::eOutNormal, &normal_info));
			writes.emplace_back(d->makeWrite(0, RtxBindings::eOutDepth, &depth_info));
			writes.emplace_back(d->makeWrite(0, RtxBindings::eOutDisparity, &discrep_info));
			writes.emplace_back(d->makeWrite(0, RtxBindings::eRayStats, &rtxStats_info));

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

		void updateRayCounters()
		{
			if (!m_gBuffers)
				return;

			const VkExtent2D sz = m_gBuffers->getSize();
			const uint64_t eyeW = static_cast<uint64_t>(sz.width) / 2;
			const uint64_t eyeH = static_cast<uint64_t>(sz.height);
			const uint64_t spp = static_cast<uint64_t>(m_settings.maxSamples);

			// Theoretical estimate (dominant + non-dominant same for now)
			m_estRaysR = eyeW * eyeH * spp;
			m_estRaysL = eyeW * eyeH * spp;
		}

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
#if defined(NVP_SUPPORTS_OPTIX9) || defined(NVP_SUPPORTS_OPTIX7)
			nvvk::Texture result{ m_gBuffers->getColorImage(eGBufResult), nullptr, m_gBuffers->getDescriptorImageInfo(eGBufResult) };
			nvvk::Texture albedo{ m_gBuffers->getColorImage(eGBufAlbedo), nullptr, m_gBuffers->getDescriptorImageInfo(eGBufAlbedo) };
			nvvk::Texture normal{ m_gBuffers->getColorImage(eGBufNormal), nullptr, m_gBuffers->getDescriptorImageInfo(eGBufNormal) };
			//nvvk::Texture depth{ m_gBuffers->getColorImage(eGBufDepth), nullptr, m_gBuffers->getDescriptorImageInfo(eGBufDepth) };
			//nvvk::Texture disparity{ m_gBuffers->getColorImage(eGBufDisparity), nullptr, m_gBuffers->getDescriptorImageInfo(eGBufDisparity) };
			m_denoiser->imageToBuffer(cmd, { result, albedo, normal });
#endif // NVP_SUPPORTS_OPTIX7
		}

		// #OPTIX_D
		// Copy the denoised buffer to Vulkan image
		void copyCudaImagesToVulkan(VkCommandBuffer cmd)
		{
#if defined(NVP_SUPPORTS_OPTIX9) || defined(NVP_SUPPORTS_OPTIX7)
			nvvk::Texture denoised{ m_gBuffers->getColorImage(eGbufDenoised), nullptr, m_gBuffers->getDescriptorImageInfo(eGbufDenoised) };
			m_denoiser->bufferToImage(cmd, &denoised);
#endif // NVP_SUPPORTS_OPTIX7
		}

		// #OPTIX_D
		// Invoke the Optix denoiser
		void denoiseImage()
		{
#if defined(NVP_SUPPORTS_OPTIX9) || defined(NVP_SUPPORTS_OPTIX7)
			m_denoiser->denoiseImageBuffer(m_fenceValue, m_blendFactor);
#endif // NVP_SUPPORTS_OPTIX7
		}

		// #OPTIX_D
		// Determine which image will be displayed, the original from ray tracer or the denoised one
		bool showDenoisedImage() const
		{
			/*if (g_enableXR)
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
			if (m_settings.logFile.is_open())
			{
				m_settings.logFile.close();
			}

			m_alloc->destroy(m_bFrameInfo);
			m_alloc->destroy(m_bRayStats);
			m_alloc->destroy(m_bRayStatsReadback);

			if (m_vrFence != VK_NULL_HANDLE)
			{
				vkDestroyFence(m_device, m_vrFence, nullptr);
				m_vrFence = VK_NULL_HANDLE;
			}
			for (auto& f : m_commandFrames)
			{
				vkFreeCommandBuffers(m_device, f.cmdPool, 2, f.cmdBuffer);
				vkDestroyCommandPool(m_device, f.cmdPool, nullptr);
			}
			m_gBuffers.reset();

			m_rtxPipe.destroy(m_device);
			m_rtxSet->deinit();
			m_sceneSet->deinit();
			m_sbt->destroy();
			m_picker->destroy();
#if defined(NVP_SUPPORTS_OPTIX9) || defined(NVP_SUPPORTS_OPTIX7)
			m_denoiser->destroy();
#endif // NVP_SUPPORTS_OPTIX7 || NVP_SUPPORTS_OPTIX9
		}

		//--------------------------------------------------------------------------------------------------
		//
		//
		nvvkhl::Application* m_app{ nullptr };
		std::unique_ptr<nvvk::DebugUtil> m_dutil;
		std::unique_ptr<AllocVma> m_alloc;

		glm::vec2 m_viewSize = { 1, 1 };
		VkDevice m_device = VK_NULL_HANDLE;                          // Convenient
		VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;          // Convenient
		std::unique_ptr<nvvkhl::GBuffer> m_gBuffers;                 // G-Buffers: color + depth
		std::unique_ptr<nvvk::DescriptorSetContainer> m_rtxSet;      // Descriptor set
		std::unique_ptr<nvvk::DescriptorSetContainer> m_sceneSet;    // Descriptor set

		// Resources
		nvvk::Buffer m_bFrameInfo;
		nvvk::Buffer m_bRayStats;         // STORAGE + TRANSFER src/dst (device local)
		nvvk::Buffer m_bRayStatsReadback; // TRANSFER dst (host visible)

		// Pipeline
		PushConstant m_pushConst{}; // Information sent to the shader
		PipelineContainer m_rtxPipe;
		int m_frame{ -1 };
		FrameInfo m_frameInfo;
		VkFence m_vrFence = VK_NULL_HANDLE;

		// Estimated (theoretical) counters computed from resolution & spp
		uint64_t m_estRaysR{ 0 };
		uint64_t m_estRaysL{ 0 };

		// Measured counters read back from GPU
		uint64_t m_measuredRaysR{ 0 };
		uint64_t m_measuredRaysL{ 0 };
		bool     m_hasMeasuredRays{ false };

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



#if defined(NVP_SUPPORTS_OPTIX9) || defined(NVP_SUPPORTS_OPTIX7)
		std::unique_ptr<DenoiserOptix> m_denoiser;
		uint64_t m_fenceValue{ 0U };
#endif // NVP_SUPPORTS_OPTIX7 || NVP_SUPPORTS_OPTIX9
		float m_blendFactor = 0.0f;
		float m_middleRadius = 60.0f;
		// XR related
		// g_xrViews
		// m_xrProjCached and m_cachedLeftProj/m_cachedRightProj
		glm::mat4 m_cachedLeftProj{ 1.0f };
		glm::mat4 m_cachedRightProj{ 1.0f };
		glm::mat4 m_cachedLeftProjInv{ 1.0f };
		glm::mat4 m_cachedRightProjInv{ 1.0f };
		bool      m_xrProjCached{ false };
		std::chrono::steady_clock::time_point m_blackScreenUntil;


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
	g_enableXR = initializeOpenXR(context);

	if (g_enableXR)
	{
		try
		{
			getSystemOpenXR();
			createOpenXRSwapchain();
			updateHeadsetDisplayInfo();
			std::cout << "\n=== OpenXR VR Mode Enabled ===" << std::endl;
			std::cout << "Swapchain created with " << g_openXRState.swapchainImages.size() << " images" << std::endl;
			std::cout << "Resolution: " << systemProperties.graphicsProperties.maxSwapchainImageWidth
				<< "x" << systemProperties.graphicsProperties.maxSwapchainImageHeight << std::endl;

		}
		catch (const std::exception& e)
		{
			std::cerr << "OpenXR post-init exception: " << e.what() << std::endl;
			std::cerr << "Falling back to desktop mode." << std::endl;
			g_enableXR = false;
		}
	}
	else
	{
		std::cout << "\n=== OpenXR initialization failed, falling back to Desktop Mode ===" << std::endl;
	}


	// Debug output to verify mode
	std::cout << "\n=== Final Mode: " << (g_enableXR ? "VR MODE" : "DESKTOP MODE") << " ===" << std::endl;

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
	//std::string scn_file = nvh::findFile(R"(media/sponza/glTF/Sponza.gltf)", default_search_paths, true);
	//std::string scn_file = nvh::findFile(R"(media/scenes/ABeautifulGame/glTF/ABeautifulGame.gltf)", default_search_paths, true);

	//optixDenoiser->onFileDrop(scn_file.c_str());
	//scn_file = nvh::findFile(R"(media/cube.gltf)", default_search_paths, true);
	//optixDenoiser->onFileDrop(scn_file.c_str());

	//CameraManip.setLookat(
	//	glm::vec3(0.0f, 1.6f, 0.0f),   // eye:    center of atrium, standing eye height
	//	glm::vec3(10.0f, 1.6f, 0.0f),  // center: looking down the long axis (+X)
	//	glm::vec3(0.0f, 1.0f, 0.0f),   // up
	//	true                            // instant (no animation)
	//);
	CameraManip.setFov(104.0f); // Estimated from Quest 3s horizontal

	// Load HDR
	//std::string hdr_file = nvh::findFile(R"(media/hdr/autumn_field_1k.hdr)", default_search_paths, true); // (180, 142, -88)
	//std::string hdr_file = nvh::findFile(R"(media/hdr/autumn_hilly_field_1k.hdr)", default_search_paths, true); // (110, 180, -96 )
	//std::string hdr_file = nvh::findFile(R"(media/hdr/golden_gate_hills_1k.hdr)", default_search_paths, true); // (-53, 151, 2)
	std::string hdr_file = nvh::findFile(R"(media/hdr/qwantani_noon_puresky_1k.hdr)", default_search_paths, true); // good (112, 49, 158)
	//std::string hdr_file = nvh::findFile(R"(media/hdr/spruit_sunrise_1k.hdr)", default_search_paths, true); //better (-4.6, 35, 121) 


	optixDenoiser->onFileDrop(hdr_file.c_str());

	// Run as fast as possible
	app->setVsync(false);

	app->run();

	// Cleanup OpenXR resources
	if (g_enableXR)
	{
		g_openXRState.cleanup();
	}
	optixDenoiser.reset();
	app.reset();

	return 0;
}
