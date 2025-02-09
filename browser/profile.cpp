#include "askgloom/browser/profile.hpp"
#include <filesystem>
#include <fstream>
#include <json/json.hpp>
#include <stdexcept>

namespace askgloom {

namespace fs = std::filesystem;
using json = nlohmann::json;

Profile::Profile(const std::string& profile_path)
    : m_profile_path(profile_path),
      m_loaded(false) {
}

bool Profile::load() {
    try {
        if (!fs::exists(m_profile_path)) {
            if (!createDefaultProfile()) {
                return false;
            }
        }

        loadPreferences();
        loadCookies();
        loadExtensions();

        m_loaded = true;
        return true;
    } catch (const std::exception& e) {
        // Logger::error("Failed to load profile: {}", e.what());
        return false;
    }
}

bool Profile::save() {
    try {
        if (!m_loaded) {
            throw std::runtime_error("Profile not loaded");
        }

        savePreferences();
        saveCookies();
        saveExtensions();

        return true;
    } catch (const std::exception& e) {
        // Logger::error("Failed to save profile: {}", e.what());
        return false;
    }
}

bool Profile::createDefaultProfile() {
    try {
        fs::create_directories(m_profile_path);
        
        // Create default profile structure
        fs::create_directories(getPreferencesPath());
        fs::create_directories(getCookiesPath());
        fs::create_directories(getExtensionsPath());

        // Initialize with default settings
        json default_prefs = {
            {"browser", {
                {"window_size", {
                    {"width", 1920},
                    {"height", 1080}
                }},
                {"startup_page", "about:blank"},
                {"download_path", (fs::path(m_profile_path) / "downloads").string()}
            }},
            {"privacy", {
                {"clear_on_exit", false},
                {"block_third_party_cookies", true}
            }}
        };

        std::ofstream prefs_file(getPreferencesPath() / "preferences.json");
        prefs_file << default_prefs.dump(4);

        return true;
    } catch (const std::exception& e) {
        // Logger::error("Failed to create default profile: {}", e.what());
        return false;
    }
}

void Profile::loadPreferences() {
    auto prefs_path = getPreferencesPath() / "preferences.json";
    if (!fs::exists(prefs_path)) {
        throw std::runtime_error("Preferences file not found");
    }

    std::ifstream prefs_file(prefs_path);
    json prefs = json::parse(prefs_file);
    
    m_preferences = prefs;
}

void Profile::loadCookies() {
    auto cookies_path = getCookiesPath() / "cookies.db";
    if (!fs::exists(cookies_path)) {
        return; // New profile, no cookies yet
    }

    // Implement SQLite cookie database loading
    // m_cookies = loadCookieDatabase(cookies_path);
}

void Profile::loadExtensions() {
    for (const auto& entry : fs::directory_iterator(getExtensionsPath())) {
        if (entry.is_directory()) {
            loadExtension(entry.path());
        }
    }
}

void Profile::loadExtension(const fs::path& ext_path) {
    auto manifest_path = ext_path / "manifest.json";
    if (!fs::exists(manifest_path)) {
        return;
    }

    std::ifstream manifest_file(manifest_path);
    json manifest = json::parse(manifest_file);
    
    Extension ext;
    ext.id = manifest["id"].get<std::string>();
    ext.name = manifest["name"].get<std::string>();
    ext.version = manifest["version"].get<std::string>();
    ext.path = ext_path.string();

    m_extensions.push_back(ext);
}

fs::path Profile::getPreferencesPath() const {
    return fs::path(m_profile_path) / "preferences";
}

fs::path Profile::getCookiesPath() const {
    return fs::path(m_profile_path) / "cookies";
}

fs::path Profile::getExtensionsPath() const {
    return fs::path(m_profile_path) / "extensions";
}

bool Profile::isLoaded() const {
    return m_loaded;
}

const std::vector<Extension>& Profile::getExtensions() const {
    return m_extensions;
}

json Profile::getPreferences() const {
    return m_preferences;
}

} // namespace askgloom