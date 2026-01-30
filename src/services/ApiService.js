/**
 * API Service for connecting React Native to Python backend
 */

// Backend URL - change this to your server IP when running on physical device
const API_BASE_URL = __DEV__
    ? 'http://localhost:8000'  // Development
    : 'http://your-production-server.com'; // Production

/**
 * Helper to perform fetch with a timeout
 */
const fetchWithTimeout = async (url, options = {}, timeout = 15000) => {
    const controller = new AbortController();
    const id = setTimeout(() => controller.abort(), timeout);

    try {
        const response = await fetch(url, {
            ...options,
            signal: controller.signal
        });
        clearTimeout(id);
        return response;
    } catch (error) {
        clearTimeout(id);
        throw error;
    }
};

export const analyzeAudio = async (audioUri) => {
    try {
        // Create form data
        const formData = new FormData();
        formData.append('audio', {
            uri: audioUri,
            type: 'audio/wav',
            name: 'recording.wav'
        });

        // 25 second timeout for heavy audio analysis
        const response = await fetchWithTimeout(`${API_BASE_URL}/analyze`, {
            method: 'POST',
            body: formData,
        }, 25000);

        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }

        const result = await response.json();

        return {
            success: true,
            data: {
                diagnosis: result.class_name,
                classId: result.class_id,
                confidence: result.confidence,
                severity: result.severity_score,
                esiLevel: result.esi_level,
                esiName: result.esi_name,
                recommendation: result.recommendation,
                probabilities: result.all_probabilities,
                heartRate: result.heart_rate
            }
        };
    } catch (error) {
        console.error('API Error:', error);
        return {
            success: false,
            error: error.name === 'AbortError' ? 'Analysis timed out' : error.message
        };
    }
};

export const checkBackendHealth = async () => {
    try {
        const response = await fetchWithTimeout(`${API_BASE_URL}/health`, {
            method: 'GET',
        }, 4000); // 4 second timeout for health check

        if (response.ok) {
            const data = await response.json();
            return {
                available: true,
                modelLoaded: data.model_loaded
            };
        }
        return { available: false };
    } catch (error) {
        return { available: false };
    }
};

export default {
    analyzeAudio,
    checkBackendHealth
};
