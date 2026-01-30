/**
 * API Service for connecting React Native to Python backend
 */

// Backend URL - change this to your server IP when running on physical device
const API_BASE_URL = __DEV__
    ? 'http://192.168.1.161:8000'  // Use PC IP for Phone (Expo Go)
    : 'http://your-production-server.com'; // Production

export const analyzeAudio = async (audioUri) => {
    try {
        // Create form data
        const formData = new FormData();
        formData.append('audio', {
            uri: audioUri,
            type: 'audio/wav',
            name: 'recording.wav'
        });

        const response = await fetch(`${API_BASE_URL}/analyze`, {
            method: 'POST',
            body: formData,
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        });

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
            error: error.message
        };
    }
};

export const checkBackendHealth = async () => {
    try {
        const response = await fetch(`${API_BASE_URL}/health`, {
            method: 'GET',
            timeout: 5000
        });

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
