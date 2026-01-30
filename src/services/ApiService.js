/**
 * API Service for connecting React Native to Python backend
 */

import Constants from 'expo-constants';

// Dynamically determine the backend URL based on the Expo host
const getApiBaseUrl = () => {
    if (__DEV__) {
        const hostUri = Constants.expoConfig?.hostUri;
        if (hostUri) {
            // Remove the port from the hostUri (e.g., "192.168.1.5:8081" -> "192.168.1.5")
            const ipAddress = hostUri.split(':')[0];
            return `http://${ipAddress}:8000`;
        }
        return 'http://127.0.0.1:8000'; // Fallback for simulator if hostUri is missing
    }
    return 'http://your-production-server.com';
};

const API_BASE_URL = getApiBaseUrl();

/**
 * Helper to perform fetch with a timeout
 */
const fetchWithTimeout = async (url, options = {}, timeout = 15000) => {
    console.log(`[API] Fetching ${url} with timeout ${timeout}ms...`);
    const controller = new AbortController();
    const id = setTimeout(() => controller.abort(), timeout);

    try {
        const response = await fetch(url, {
            ...options,
            signal: controller.signal
        });
        clearTimeout(id);
        console.log(`[API] Response from ${url}: ${response.status}`);
        return response;
    } catch (error) {
        clearTimeout(id);
        console.error(`[API] Error fetching ${url}:`, error);
        throw error;
    }
};

export const analyzeAudio = async (audioUri) => {
    try {
        console.log('[API] Starting audio analysis...');
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
        console.log('[API] Analysis success:', result.class_name);

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
        console.log('[API] Checking backend health...');
        const response = await fetchWithTimeout(`${API_BASE_URL}/health`, {
            method: 'GET',
        }, 2000); // Reduced to 2s for faster failover

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
