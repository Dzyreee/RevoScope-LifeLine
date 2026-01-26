import React, { createContext, useContext, useState, useEffect } from 'react';
import * as DB from '../services/DatabaseService';

const AppContext = createContext();

export const AppProvider = ({ children }) => {
    const [dashboardStats, setDashboardStats] = useState({ total: 0, critical: 0, monitoring: 0, normal: 0 });
    const [patients, setPatients] = useState([]);
    const [loading, setLoading] = useState(true);

    const refreshDashboard = async () => {
        try {
            setLoading(true);
            const stats = await DB.getDashboardStats();
            const allPatients = await DB.getPatients();
            setDashboardStats(stats);
            setPatients(allPatients);
            setLoading(false);
        } catch (e) {
            console.error("Dashboard refresh error", e);
            setLoading(false);
        }
    };

    useEffect(() => {
        (async () => {
            await DB.initDB();
            await refreshDashboard();
        })();
    }, []);

    const resetDatabase = async () => {
        try {
            await DB.resetDB();
            await refreshDashboard();
        } catch (e) {
            console.error("Reset database error", e);
        }
    };

    return (
        <AppContext.Provider value={{
            dashboardStats,
            patients,
            loading,
            refreshDashboard,
            resetDatabase,
            createPatient: DB.addPatient,
            updatePatient: DB.updatePatient,
            recordScan: DB.addScan,
            deletePatient: DB.deletePatient,
            getHistory: DB.getPatientHistory
        }}>
            {children}
        </AppContext.Provider>
    );
};

export const useApp = () => useContext(AppContext);
