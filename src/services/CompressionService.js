import pako from 'pako';
import { decode as atob, encode as btoa } from 'base-64';

// Minification Map (Long keys -> Short keys)
const KEY_MAP = {
    // Patient Info
    id: 'i',
    name: 'n',
    age: 'a',
    gender: 'g',
    condition: 'c',
    notes: 'nt',
    timestamp: 'ts',

    // Vitals & Medical
    heart_rate: 'h',
    respiratory_rate: 'r',
    sp02: 's',
    temperature: 'tm',
    systolic_bp: 'sb',
    diastolic_bp: 'db',

    // Triage / Analysis
    esi_level: 'e',
    severity_score: 'sv',
    confidence_score: 'cs',
    diagnosis: 'd',
    recommendation: 'rc',
    status: 'st'
};

// Reverse Map for Decompression
const REVERSE_KEY_MAP = Object.entries(KEY_MAP).reduce((acc, [key, val]) => {
    acc[val] = key;
    return acc;
}, {});

// Value Enums (String -> Int)
const GENDER_MAP = { 'Male': 0, 'Female': 1, 'Other': 2, 'Unknown': 3 };
const REVERSE_GENDER_MAP = { 0: 'Male', 1: 'Female', 2: 'Other', 3: 'Unknown' };

const STATUS_MAP = { 'Normal': 0, 'Monitoring': 1, 'Critical': 2 };
const REVERSE_STATUS_MAP = { 0: 'Normal', 1: 'Monitoring', 2: 'Critical' };

/**
 * Minify a single object or array of objects based on key maps
 */
const minifyData = (data) => {
    if (Array.isArray(data)) {
        return data.map(item => minifyData(item));
    }

    if (typeof data === 'object' && data !== null) {
        const minified = {};
        for (const key in data) {
            const shortKey = KEY_MAP[key] || key; // Use short key if exists, else original
            let value = data[key];

            // Value optimizations
            if (key === 'gender' && GENDER_MAP[value] !== undefined) value = GENDER_MAP[value];
            if (key === 'status' && STATUS_MAP[value] !== undefined) value = STATUS_MAP[value];

            // Recursive minification for nested objects
            if (typeof value === 'object') {
                value = minifyData(value);
            }

            minified[shortKey] = value;
        }
        return minified;
    }
    return data;
};

/**
 * Unminify data back to original keys and values
 */
const unminifyData = (data) => {
    if (Array.isArray(data)) {
        return data.map(item => unminifyData(item));
    }

    if (typeof data === 'object' && data !== null) {
        const original = {};
        for (const key in data) {
            const longKey = REVERSE_KEY_MAP[key] || key;
            let value = data[key];

            // Value restorations
            if (longKey === 'gender' && REVERSE_GENDER_MAP[value] !== undefined) value = REVERSE_GENDER_MAP[value];
            if (longKey === 'status' && REVERSE_STATUS_MAP[value] !== undefined) value = REVERSE_STATUS_MAP[value];

            if (typeof value === 'object') {
                value = unminifyData(value);
            }

            original[longKey] = value;
        }
        return original;
    }
    return data;
};

/**
 * Compress: Minify -> JSON String -> Gzip -> Base64
 */
export const compressPacket = (data) => {
    try {
        const minified = minifyData(data);
        const jsonString = JSON.stringify(minified);
        const compressed = pako.gzip(jsonString); // Uint8Array

        // Convert Uint8Array to Binary String for btoa
        let binary = '';
        const len = compressed.byteLength;
        for (let i = 0; i < len; i++) {
            binary += String.fromCharCode(compressed[i]);
        }

        return btoa(binary);
    } catch (e) {
        console.error('Compression failed:', e);
        return null;
    }
};

/**
 * Decompress: Base64 -> Gzip -> JSON Parse -> Unminify
 */
export const decompressPacket = (encodedData) => {
    try {
        const binary = atob(encodedData);
        const charData = binary.split('').map(x => x.charCodeAt(0));
        const binData = new Uint8Array(charData);

        const inflated = pako.ungzip(binData, { to: 'string' });
        const minified = JSON.parse(inflated);

        return unminifyData(minified);
    } catch (e) {
        console.error('Decompression failed:', e);
        // Fallback: maybe it's raw JSON?
        try {
            return JSON.parse(encodedData);
        } catch (e2) {
            return null;
        }
    }
};
