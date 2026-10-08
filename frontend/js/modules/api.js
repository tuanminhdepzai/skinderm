/**
 * api.js - Backend Communication Module
 */

import state from './state.js';

const API_BASE_URL = (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' || window.location.protocol === 'file:')
    ? 'http://localhost:8080/api'
    : 'https://codebykdvn--skinderm-backend-fastapi-app.modal.run/api';

const API = {
    /**
     * Fetch trending dermatology topics and blogs
     */
    async getExploreContent(lang = 'vi') {
        try {
            const response = await fetch(`${API_BASE_URL}/explore?language=${lang}`);
            if (!response.ok) throw new Error("Failed to fetch explore content");
            return await response.json();
        } catch (err) {
            console.error("Explore API Error:", err);
            return { status: 'error', data: [] };
        }
    },

    /**
     * Get user scan history
     */
    async getHistory(idToken) {
        if (!idToken) return { status: 'error', message: 'No token provided' };
        
        try {
            const formData = new FormData();
            formData.append('authorization', `Bearer ${idToken}`);

            const response = await fetch(`${API_BASE_URL}/history`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error("Failed to fetch history");
            return await response.json();
        } catch (err) {
            console.error("History API Error:", err);
            return { status: 'error', data: [] };
        }
    },

    /**
     * Analyze image
     */
    async analyzeImage(imageBlob, idToken, weatherData, prevScan = null) {
        try {
            const formData = new FormData();
            formData.append('file', imageBlob, 'scan.jpg');
            
            if (idToken) {
                formData.append('authorization', `Bearer ${idToken}`);
            }

            if (weatherData) {
                formData.append('uv_index', weatherData.uvIndex);
                formData.append('temperature', weatherData.temp);
                formData.append('location', weatherData.location);
            }

            if (prevScan && prevScan.details) {
                formData.append('prev_abcde', JSON.stringify(prevScan.details.abcde));
                formData.append('prev_risk', prevScan.details.risk_score || prevScan.risk);
            }

            const response = await fetch(`${API_BASE_URL}/analyze`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error("Analysis API Error");
            return await response.json();
        } catch (err) {
            console.error("Analyze API Error:", err);
            throw err;
        }
    },

    /**
     * Chat interaction with AI Doctor
     */
    async startChatStream(text, model, context, signal) {
        return await fetch(`${API_BASE_URL}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            signal: signal,
            body: JSON.stringify({
                message: text,
                model: model,
                context: context
            })
        });
    },

    /**
     * Compare two scans using AI
     */
    async compareScans(scan1, scan2) {
        try {
            const response = await fetch(`${API_BASE_URL}/compare`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ scan1, scan2 })
            });
            if (!response.ok) throw new Error("Failed to compare scans");
            return await response.json();
        } catch (err) {
            console.error("Compare API Error:", err);
            return { status: 'error', message: err.message };
        }
    },
    
    /**
     * Upload user avatar
     */
    async uploadAvatar(file, idToken) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('authorization', `Bearer ${idToken}`);

        const response = await fetch(`${API_BASE_URL}/upload-avatar`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) throw new Error("Upload failed");
        return await response.json();
    }
};

export default API;
