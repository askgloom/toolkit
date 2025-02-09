#include "askgloom/browser/browser.hpp"
#include <stdexcept>
#include <chrono>
#include <thread>

namespace askgloom {

Browser::Browser(const BrowserConfig& config) 
    : m_profile(config.profile_path),
      m_headless(config.headless),
      m_initialized(false) {
    initialize();
}

Browser::~Browser() {
    if (m_initialized) {
        cleanup();
    }
}

void Browser::initialize() {
    if (m_initialized) {
        return;
    }

    try {
        // Load browser profile
        if (!m_profile.load()) {
            throw std::runtime_error("Failed to load browser profile");
        }

        // Initialize browser instance
        if (!initializeBrowserInstance()) {
            throw std::runtime_error("Failed to initialize browser instance");
        }

        m_initialized = true;
    } catch (const std::exception& e) {
        cleanup();
        throw std::runtime_error(std::string("Browser initialization failed: ") + e.what());
    }
}

bool Browser::initializeBrowserInstance() {
    // Platform-specific browser initialization
    #ifdef _WIN32
        return initializeWindowsBrowser();
    #elif __linux__
        return initializeLinuxBrowser();
    #elif __APPLE__
        return initializeMacBrowser();
    #else
        throw std::runtime_error("Unsupported platform");
    #endif
}

void Browser::navigate(const std::string& url) {
    if (!m_initialized) {
        throw std::runtime_error("Browser not initialized");
    }

    try {
        // Perform navigation
        performNavigation(url);
        
        // Wait for page load
        waitForPageLoad();
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Navigation failed: ") + e.what());
    }
}

void Browser::waitForPageLoad(int timeout_ms) {
    auto start = std::chrono::steady_clock::now();
    
    while (true) {
        if (isPageLoaded()) {
            return;
        }

        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start);
        
        if (elapsed.count() > timeout_ms) {
            throw std::runtime_error("Page load timeout");
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

bool Browser::isPageLoaded() const {
    // Check various page load indicators
    return checkDocumentReady() && 
           checkNetworkIdle() && 
           checkRenderComplete();
}

void Browser::cleanup() {
    try {
        if (m_initialized) {
            // Close all windows
            closeAllWindows();
            
            // Clean up browser instance
            cleanupBrowserInstance();
            
            m_initialized = false;
        }
    } catch (const std::exception& e) {
        // Log error but don't throw from destructor
        // Logger::error("Browser cleanup failed: {}", e.what());
    }
}

std::string Browser::getCurrentUrl() const {
    if (!m_initialized) {
        throw std::runtime_error("Browser not initialized");
    }
    return m_current_url;
}

bool Browser::isInitialized() const {
    return m_initialized;
}

} // namespace askgloom