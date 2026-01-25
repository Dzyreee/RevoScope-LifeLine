import React, { createContext, useState, useEffect, useContext } from 'react';
import { initDB, getPatients, addPatient as addPatientToDB, updatePatientCategory } from '../services/DatabaseService';

const AppContext = createContext();

export const AppProvider = ({ children }) => {
    const [patients, setPatients] = useState([]);
    const [audioSource, setAudioSource] = useState('internal'); // 'internal' | 'external'
    const [isLoading, setIsLoading] = useState(true);

    useEffect(() => {
        const loadData = async () => {
            try {
                await initDB();
                const storedPatients = await getPatients();
                setPatients(storedPatients);
            } catch (error) {
                console.error("Failed to init DB or load patients", error);
            } finally {
                setIsLoading(false);
            }
        };
        loadData();
    }, []);

    const refreshPatients = async () => {
        try {
            const storedPatients = await getPatients();
            setPatients(storedPatients);
        } catch (error) {
            console.error("Failed to refresh patients", error);
        }
    };

    const notifyScanComplete = async (patientId, diagnosticResult) => {
        // Logic to update patient with scan result could go here
        // For now we might just want to refresh
        await refreshPatients();
    };

    const addNewPatient = async (patientData) => {
        try {
            await addPatientToDB(patientData);
            await refreshPatients();
        } catch (error) {
            console.error("Failed to add patient", error);
            throw error;
        }
    }

    const movePatientToCategory = async (patientId, category) => {
        try {
            await updatePatientCategory(patientId, category);
            await refreshPatients();
        } catch (error) {
            console.error("Failed to move patient", error);
        }
    }

    return (
        <AppContext.Provider
            value={{
                patients,
                isLoading,
                addNewPatient,
                refreshPatients,
                audioSource,
                setAudioSource,
                movePatientToCategory
            }}
        >
            {children}
        </AppContext.Provider>
    );
};

export const useApp = () => useContext(AppContext);
