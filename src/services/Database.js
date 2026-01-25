// In a real device scenario, we would use react-native-sqlite-storage
// import SQLite from 'react-native-sqlite-storage';

// Store for the session
const mockDB = [];

export const getPatients = async () => {
    // Simulate async DB call
    return new Promise((resolve) => {
        setTimeout(() => resolve([...mockDB]), 200);
    });
};

export const saveScan = async (patientDetails, diagnosis, confidence, status, spectrogramData) => {
    const newRecord = {
        id: Math.floor(Math.random() * 100000).toString(),
        timestamp: new Date().toISOString(),
        ...patientDetails,
        status,
        diagnosis,
        confidence,
        spectrogramData // Store the array of data for playback
    };
    // Add to beginning
    mockDB.unshift(newRecord);
    return newRecord;
};
