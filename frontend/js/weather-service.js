/**
 * WeatherService.js
 * Handles fetching weather and UV data from Open-Meteo API.
 */

class WeatherService {
    constructor() {
        this.baseUrl = "https://api.open-meteo.com/v1/forecast";
    }

    /**
     * Fast client-side location detection via IP (works without permission popups)
     */
    async detectLocationByIP() {
        try {
            const res = await fetch("https://get.geojs.io/v1/ip/geo.json");
            if (res.ok) {
                const data = await res.json();
                if (data.latitude && data.longitude) {
                    const city = data.city || data.region || "Việt Nam";
                    const country = data.country_code ? data.country_code.toUpperCase() : "VN";
                    return {
                        lat: parseFloat(data.latitude),
                        lon: parseFloat(data.longitude),
                        locationName: `${city}, ${country}`
                    };
                }
            }
        } catch (e) {
            console.warn("IP Geo detection error:", e);
        }
        return null;
    }

    /**
     * Get weather and UV data for coordinates.
     * @param {number} lat 
     * @param {number} lon 
     * @param {string|null} knownLocation
     */
    async getWeatherData(lat = 21.0285, lon = 105.8542, knownLocation = null) {
        try {
            const url = `${this.baseUrl}?latitude=${lat}&longitude=${lon}&current=temperature_2m,relative_humidity_2m,weather_code&hourly=uv_index&timezone=auto&forecast_days=1`;
            const response = await fetch(url);
            if (!response.ok) throw new Error("Weather API Error");
            
            const data = await response.json();
            
            // Get current UV index (closest hourly value)
            const hour = new Date().getHours();
            const uvIndex = (data.hourly && data.hourly.uv_index) ? (data.hourly.uv_index[hour] || 0) : 0;

            // Location Name resolution
            let locationName = knownLocation;
            if (!locationName) {
                locationName = await this.getReverseGeocoding(lat, lon);
            }
            
            return {
                temp: data.current.temperature_2m,
                humidity: data.current.relative_humidity_2m,
                weatherCode: data.current.weather_code,
                uvIndex: uvIndex,
                location: locationName || "Việt Nam"
            };
        } catch (error) {
            console.error("Failed to fetch weather:", error);
            return null;
        }
    }

    /**
     * Get human-readable location name from coordinates using client-friendly APIs.
     */
    async getReverseGeocoding(lat, lon) {
        // 1. Try BigDataCloud (CORS enabled, client-side safe, Vietnamese localized)
        try {
            const bdcUrl = `https://api.bigdatacloud.net/data/reverse-geocode-client?latitude=${lat}&longitude=${lon}&localityLanguage=vi`;
            const bdcRes = await fetch(bdcUrl);
            if (bdcRes.ok) {
                const data = await bdcRes.json();
                const city = data.city || data.locality || data.principalSubdivision;
                const country = data.countryCode || "VN";
                if (city) {
                    return `${city}, ${country}`;
                }
            }
        } catch (e) {
            console.warn("BigDataCloud geocoding failed, trying fallback:", e);
        }

        // 2. Fallback Nominatim (without forbidden User-Agent header)
        try {
            const url = `https://nominatim.openstreetmap.org/reverse?format=json&lat=${lat}&lon=${lon}&zoom=12&addressdetails=1`;
            const response = await fetch(url);
            if (response.ok) {
                const data = await response.json();
                const addr = data.address || {};
                const city = addr.city || addr.town || addr.village || addr.suburb || addr.state || addr.county;
                const country = addr.country_code ? addr.country_code.toUpperCase() : "VN";
                if (city) {
                    return `${city}, ${country}`;
                }
            }
        } catch (error) {
            console.warn("Reverse geocoding error:", error);
        }

        return "Vị trí hiện tại";
    }

    getUVAdvice(uvIndex) {
        if (uvIndex < 3) return { level: "Thấp", advice: "An toàn để ra ngoài. Cần dưỡng ẩm nhẹ.", color: "#10b981", video: "Mesh_strips_wave,_202603211318.mp4" };
        if (uvIndex < 6) return { level: "Trung bình", advice: "Nên che chắn khi ra ngoài lâu. Dùng kem chống nắng SPF 30+.", color: "#f59e0b", video: "Mesh_strips_wave,_202603211318.mp4" };
        if (uvIndex < 8) return { level: "Cao", advice: "Nguy cơ tổn thương da cao. Tránh nắng từ 10h-16h. Dùng SPF 50+.", color: "#ef4444", video: "sunny.mp4" };
        return { level: "Rất cao", advice: "Cực kỳ nguy hiểm. Hạn chế ra ngoài tối đa. Che chắn kỹ và dùng KCN mạnh.", color: "#7c3aed", video: "sunny.mp4" };
    }

    /**
     * Map Open-Meteo weather codes to a theme (video background).
     * @param {number} code 
     * @param {number} uvIndex
     */
    getWeatherTheme(code, uvIndex = 0) {
        if (uvIndex >= 6 && code <= 3) {
            return { type: "sunny", label: "Trời rất nắng", video: "sunny.mp4" };
        }
        if (code === 0 || code === 1 || code === 2) return { type: "sunny", label: "Trời nắng", video: "sunny.mp4" };
        if (code === 3) return { type: "cloudy", label: "Trời râm", video: "cloudy.mp4" };
        if ((code >= 51 && code <= 67) || (code >= 80 && code <= 82) || (code >= 95)) 
            return { type: "rainy", label: "Trời mưa", video: "rainy.mp4" };
        return { type: "dull", label: "Trời âm u", video: "cloudy.mp4" };
    }
}

export default new WeatherService();
