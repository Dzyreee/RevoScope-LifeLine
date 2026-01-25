import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { useApp } from '../context/AppContext';

export default function ImagingScreen({ route, navigation }) {
    const { patients, movePatientToCategory, refreshPatients } = useApp();
    const { patientId } = route.params || {};

    const patient = patients.find(p => p.id === patientId);

    const [isScanning, setIsScanning] = useState(false);
    const [spectrogramData, setSpectrogramData] = useState([]);
    const [result, setResult] = useState(null);

    // Simulate Spectrogram Data Stream
    useEffect(() => {
        let interval;
        if (isScanning) {
            interval = setInterval(() => {
                const newColumn = Array.from({ length: 20 }, () => Math.random());
                setSpectrogramData(prev => {
                    const newData = [...prev, newColumn];
                    if (newData.length > 30) newData.shift();
                    return newData;
                });
            }, 100);

            // Simulate Scan Completion after 5 seconds
            setTimeout(() => {
                finishScan();
            }, 5000);
        }
        return () => clearInterval(interval);
    }, [isScanning]);

    const startScan = () => {
        setIsScanning(true);
        setResult(null);
        setSpectrogramData([]);
    };

    const finishScan = async () => {
        setIsScanning(false);
        // Mock Result Logic
        const outcomes = [
            { d: 'Normal Breath Sounds', s: 'P3', c: 0.95 },
            { d: 'Crackles/Wheezing', s: 'P2', c: 0.89 },
            { d: 'Absent Breath Sounds', s: 'P1', c: 0.91 }
        ];
        // Random outcome
        const outcome = outcomes[Math.floor(Math.random() * outcomes.length)];

        setResult(outcome);

        // Update Global State / DB
        if (patientId) {
            // In a real app we'd save the specific scan result/image too.
            // For now we just update their category and 'last scan result'
            // We need to implement updating 'last_scan_result' in DB service effectively, 
            // but `updatePatientCategory` handles the category. 
            // Ideally we'd have a `updatePatientScanResult` method.
            // For this demo, moving category is the key requirement.
            await movePatientToCategory(patientId, outcome.s);
        }
    };

    const renderSpectrogram = () => {
        if (spectrogramData.length === 0 && !isScanning) return null;
        return (
            <View style={styles.spectrogramContainer}>
                {spectrogramData.map((col, i) => (
                    <View key={i} style={styles.spectrogramColumn}>
                        {col.map((val, j) => (
                            <View key={j} style={[styles.spectrogramCell, { backgroundColor: `rgba(59, 130, 246, ${val})` }]} />
                        ))}
                    </View>
                ))}
            </View>
        );
    };

    if (!patient) {
        return (
            <View style={styles.container}>
                <Text style={{ color: 'white', alignSelf: 'center', marginTop: 100 }}>Patient not found</Text>
            </View>
        )
    }

    return (
        <View style={styles.container}>
            <View style={styles.displayArea}>
                <Text style={styles.patientName}>{patient.name}</Text>
                {renderSpectrogram()}
                {isScanning && <Text style={styles.scanningText}>Acquiring Acoustic Data...</Text>}
            </View>

            <View style={styles.controls}>
                {result ? (
                    <View style={styles.resultContainer}>
                        <Text style={styles.resultTitle}>Diagnosis Complete</Text>
                        <Text style={[
                            styles.resultStatus,
                            { color: result.s === 'P1' ? '#D32F2F' : result.s === 'P2' ? '#FBC02D' : '#388E3C' }
                        ]}>
                            {result.s === 'P1' ? 'CRITICAL' : result.s === 'P2' ? 'MODERATE' : 'STABLE'}
                        </Text>
                        <Text style={styles.diagnosisText}>{result.d}</Text>
                        <Text style={styles.confidence}>Confidence: {(result.c * 100).toFixed(1)}%</Text>

                        <TouchableOpacity style={styles.button} onPress={() => navigation.goBack()}>
                            <Text style={styles.buttonText}>Return to Dashboard</Text>
                        </TouchableOpacity>
                    </View>
                ) : (
                    <TouchableOpacity
                        style={[styles.button, isScanning && styles.buttonDisabled]}
                        onPress={startScan}
                        disabled={isScanning}
                    >
                        <Text style={styles.buttonText}>{isScanning ? 'Scanning...' : 'Start Diagnostic Scan'}</Text>
                    </TouchableOpacity>
                )}
            </View>
        </View>
    );
}

const styles = StyleSheet.create({
    container: { flex: 1, backgroundColor: '#09090b' },
    patientName: {
        position: 'absolute',
        top: 20,
        color: '#fff',
        fontSize: 18,
        fontWeight: 'bold',
        zIndex: 10,
    },
    scanningText: { color: '#3b82f6', marginTop: 20, fontSize: 16, fontWeight: '600', letterSpacing: 1 },
    diagnosisText: { color: '#fff', fontSize: 18, marginBottom: 8, textAlign: 'center' },
    displayArea: {
        flex: 2,
        justifyContent: 'center',
        alignItems: 'center',
        backgroundColor: '#000',
        borderBottomWidth: 1,
        borderBottomColor: '#27272a'
    },
    spectrogramContainer: {
        flexDirection: 'row',
        height: 200,
        alignItems: 'flex-end',
        justifyContent: 'center',
        gap: 2
    },
    spectrogramColumn: {
        flexDirection: 'column-reverse',
        gap: 1
    },
    spectrogramCell: {
        width: 8,
        height: 8,
        borderRadius: 1
    },
    controls: { flex: 1, padding: 20, justifyContent: 'center' },
    button: {
        backgroundColor: '#3b82f6',
        padding: 16,
        borderRadius: 12,
        alignItems: 'center',
        marginTop: 20,
    },
    buttonDisabled: { backgroundColor: '#1e3a8a' },
    buttonText: { color: '#fff', fontSize: 18, fontWeight: 'bold' },
    resultContainer: { alignItems: 'center', gap: 10 },
    resultTitle: { color: '#fff', fontSize: 20 },
    resultStatus: { fontSize: 24, fontWeight: 'bold' },
    confidence: { color: '#a1a1aa' }
});
