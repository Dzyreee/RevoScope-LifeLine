import React, { useState, useCallback, useEffect, useRef } from 'react';
import { View, Text, ScrollView, TouchableOpacity, Modal, Animated, Image, Alert, StyleSheet, Dimensions, Switch, TextInput, ActivityIndicator } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
// import { useFocusEffect } from '@react-navigation/native';
import { StatusBar } from 'expo-status-bar';
import { Ionicons } from '@expo/vector-icons';
import { Audio } from 'expo-av';
import { useApp } from '../context/AppContext';
import StatusFilterCard from '../components/StatusFilterCard';
import PatientQueueCard from '../components/PatientQueueCard';
// import QRCode from 'react-native-qrcode-svg'; // Removed for demo
// import { CameraView, useCameraPermissions } from 'expo-camera'; // Removed for demo
import { compressPacket, decompressPacket } from '../services/CompressionService';
import { addPatient } from '../services/DatabaseService';

// Import logo
const logoImage = require('../assets/logo.png');

export default function DashboardScreen({ navigation }) {
    const { dashboardStats, patients, refreshDashboard, resetDatabase, isTestingMode, setIsTestingMode } = useApp();
    const [filter, setFilter] = useState(null);
    const [showSettings, setShowSettings] = useState(false);
    const [showSync, setShowSync] = useState(false);
    const [showDevMenu, setShowDevMenu] = useState(false);
    const [audioLevel, setAudioLevel] = useState(0);
    const recordingRef = useRef(null);
    const audioLevelAnim = useRef(new Animated.Value(0)).current;
    const [isOffline, setIsOffline] = useState(false);

    // Sync Simulation States
    const [isScanning, setIsScanning] = useState(false);
    const [foundPeers, setFoundPeers] = useState([]);
    const [isSyncing, setIsSyncing] = useState(false);
    const [syncProgress, setSyncProgress] = useState(0);
    const [syncLog, setSyncLog] = useState([]);

    // Mock Peer Data
    const MOCK_PEER = { id: 'dev_unit_04', name: 'Triage Unit #04', distance: 'Nearby' };

    const startScan = () => {
        setIsScanning(true);
        setFoundPeers([]);
        setSyncLog(prev => [...prev, 'Scanning for nearby RevoMesh nodes...']);

        // Simulate finding a peer after 2 seconds
        setTimeout(() => {
            setIsScanning(false);
            setFoundPeers([MOCK_PEER]);
            setSyncLog(prev => [...prev, `Found node: ${MOCK_PEER.name} (${MOCK_PEER.distance})`]);
        }, 2500);
    };

    const startSync = (peer) => {
        setIsSyncing(true);
        setSyncProgress(0);
        setSyncLog(prev => [...prev, `Initiating handshake with ${peer.name}...`]);

        // Simulate connection and sync progress
        let progress = 0;
        const interval = setInterval(() => {
            progress += Math.random() * 0.2;
            if (progress >= 1) {
                progress = 1;
                clearInterval(interval);
                setIsSyncing(false);
                setSyncLog(prev => [...prev, 'Sync packet received.', 'Decrypting data...', 'Merge complete.']);

                // Simulate receiving new data (optional: actually add a dummy patient)
                Alert.alert("Sync Complete", `Successfully exchanged data with ${peer.name}. Database updated.`);
                // In a real scenario, we'd merging data here. 
                // For demo, we just show success.
            }
            setSyncProgress(progress);
        }, 500);
    };


    useEffect(() => {
        const checkOfflineStatus = async () => {
            const token = await AsyncStorage.getItem('userToken');
            if (token) {
                const user = JSON.parse(token);
                setIsOffline(!!user.offline);
            }
        };
        checkOfflineStatus();
    }, []);

    useEffect(() => {
        if (navigation) {
            const unsubscribe = navigation.addListener('focus', () => {
                refreshDashboard();
            });
            return unsubscribe;
        }
    }, [navigation]);

    // Request audio permission on mount
    useEffect(() => {
        Audio.requestPermissionsAsync();
    }, []);

    // Audio level monitoring
    const startAudioMonitoring = async () => {
        try {
            await Audio.setAudioModeAsync({
                allowsRecordingIOS: true,
                playsInSilentModeIOS: true,
            });
            const { recording } = await Audio.Recording.createAsync(
                Audio.RecordingOptionsPresets.HIGH_QUALITY,
                (status) => {
                    if (status.metering !== undefined) {
                        const level = Math.max(0, Math.min(1, (status.metering + 60) / 60));
                        setAudioLevel(level);
                        Animated.timing(audioLevelAnim, {
                            toValue: level,
                            duration: 100,
                            useNativeDriver: false,
                        }).start();
                    }
                },
                100
            );
            recordingRef.current = recording;
        } catch (e) {
            console.error('Failed to start audio monitoring:', e);
        }
    };

    const stopAudioMonitoring = async () => {
        if (recordingRef.current) {
            try {
                const status = await recordingRef.current.getStatusAsync();
                if (status.isRecording) {
                    await recordingRef.current.stopAndUnloadAsync();
                }
            } catch (e) { } finally {
                recordingRef.current = null;
            }
        }
        setAudioLevel(0);
    };

    useEffect(() => {
        if (showSettings) {
            startAudioMonitoring();
        } else {
            stopAudioMonitoring();
        }
        return () => stopAudioMonitoring();
    }, [showSettings]);

    const getFilteredPatients = () => {
        if (!filter) return [];
        if (filter === 'Critical') return patients.filter(p => p.severity_score >= 70);
        if (filter === 'Monitoring') return patients.filter(p => p.severity_score >= 30 && p.severity_score < 70);
        if (filter === 'Normal') return patients.filter(p => p.severity_score < 30);
        return patients;
    };

    const filteredList = getFilteredPatients();
    const levelWidth = audioLevelAnim.interpolate({
        inputRange: [0, 1],
        outputRange: ['0%', '100%'],
    });

    const packetSizeMsg = `${(compressPacket(patients).length / 1024).toFixed(1)} KB`;

    return (
        <View className="flex-1 bg-white">
            <StatusBar style="dark" />

            {/* Header */}
            <View className="pt-16 px-6 pb-4 bg-white border-b border-gray-100 flex-row justify-between items-center h-40">
                <Image source={logoImage} style={{ width: 240, height: 90, marginTop: 10, marginLeft: -30 }} resizeMode="contain" />
                <View className="items-end justify-center h-full pt-4">
                    <View className="flex-row items-center gap-2">

                        <TouchableOpacity
                            className="h-10 w-10 bg-blue-50 rounded-full items-center justify-center shadow-sm"
                            onPress={() => navigation.navigate('Help')}
                            style={{ shadowColor: '#3B82F6', shadowOpacity: 0.1, shadowRadius: 3 }}
                        >
                            <Ionicons name="help-circle-outline" size={22} color="#3B82F6" />
                        </TouchableOpacity>
                        <TouchableOpacity
                            className="h-10 w-10 bg-indigo-50 rounded-full items-center justify-center shadow-sm border border-indigo-100"
                            onPress={() => setShowSync(true)}
                        >
                            <Ionicons name="radio-outline" size={20} color="#4F46E5" />
                        </TouchableOpacity>
                        <TouchableOpacity
                            className="h-10 w-10 bg-gray-50 rounded-full items-center justify-center border border-gray-100 shadow-sm"
                            onPress={() => setShowSettings(true)}
                        >
                            <Ionicons name="settings-outline" size={22} color="#374151" />
                        </TouchableOpacity>
                    </View>
                    {isOffline && (
                        <View className="bg-gray-100 px-3 py-1.5 rounded-full mt-3 border border-gray-200">
                            <Text className="text-[10px] font-bold text-gray-500 tracking-wider">OFFLINE</Text>
                        </View>
                    )}
                </View>
            </View>

            {/* Filter Tabs */}
            <View className="flex-row px-4 py-5">
                {[
                    { type: 'Critical', count: dashboardStats.critical },
                    { type: 'Monitoring', count: dashboardStats.monitoring },
                    { type: 'Normal', count: dashboardStats.normal }
                ].map((item) => (
                    <StatusFilterCard
                        key={item.type}
                        type={item.type}
                        count={item.count}
                        active={filter === item.type}
                        onPress={() => setFilter(filter === item.type ? null : item.type)}
                    />
                ))}
            </View>

            {/* Content Area */}
            <ScrollView className="flex-1 px-4" contentContainerStyle={{ paddingBottom: 100 }}>
                {filter ? (
                    <>
                        <TouchableOpacity
                            className="flex-row items-center mb-6 bg-gray-50 p-4 rounded-2xl border border-gray-100 shadow-sm"
                            onPress={() => setFilter(null)}
                        >
                            <Ionicons name="arrow-back" size={20} color="#374151" />
                            <Text className="text-gray-700 font-bold ml-2">Back to Dashboard Overview</Text>
                        </TouchableOpacity>

                        <Text className="text-gray-500 font-bold text-sm uppercase tracking-wider mb-4 px-2">
                            {filter} Patients ({filteredList.length})
                        </Text>
                        {filteredList.length === 0 ? (
                            <View className="items-center justify-center py-16">
                                <Ionicons name="checkmark-circle-outline" size={48} color="#9CA3AF" />
                                <Text className="text-gray-400 mt-4 text-lg">No {filter.toLowerCase()} patients</Text>
                            </View>
                        ) : (
                            filteredList.map(p => (
                                <PatientQueueCard
                                    key={p.id}
                                    patient={p}
                                    onPress={() => navigation.navigate('Detail', { patientId: p.id })}
                                />
                            ))
                        )}
                    </>
                ) : (
                    <View className="items-center justify-center py-20">
                        <View className="h-24 w-24 bg-red-50 rounded-full items-center justify-center mb-6">
                            <Ionicons name="pulse" size={48} color="#DC2626" />
                        </View>
                        <Text className="text-gray-800 text-xl font-bold text-center">Welcome to RevoScope</Text>
                        <Text className="text-gray-500 text-base mt-2 text-center px-8">
                            Select a category above to view patients,{'\n'}or add a new patient to begin.
                        </Text>
                        <TouchableOpacity
                            className="bg-red-600 px-8 py-4 rounded-xl mt-8 flex-row items-center"
                            onPress={() => navigation.navigate('Intake')}
                        >
                            <Ionicons name="person-add" size={20} color="#fff" />
                            <Text className="text-white font-bold text-lg ml-2">Add New Patient</Text>
                        </TouchableOpacity>
                    </View>
                )}
            </ScrollView>

            {/* Simulated P2P Sync Modal */}
            <Modal visible={showSync} transparent animationType="slide">
                <View className="flex-1 bg-black/60 justify-end">
                    <View className="bg-white rounded-t-3xl p-6 pb-8" style={{ height: '70%' }}>
                        <View className="flex-row justify-between items-center mb-6">
                            <View>
                                <Text className="text-2xl font-bold text-gray-800">RevoMesh Peer Sync</Text>
                                <Text className="text-sm text-indigo-600 font-bold">● P2P BLUETOOTH MESH</Text>
                            </View>
                            <TouchableOpacity onPress={() => setShowSync(false)} className="bg-gray-100 p-2 rounded-full">
                                <Ionicons name="close" size={24} color="#6B7280" />
                            </TouchableOpacity>
                        </View>

                        <Text className="text-gray-500 text-sm mb-4">
                            Seamlessly auto-sync patient records with nearby triage units without internet connection.
                        </Text>

                        <View className="bg-gray-50 rounded-2xl p-4 flex-1 border border-gray-100 relative overflow-hidden">
                            {/* Radar Scan Animation UI */}
                            {isScanning && (
                                <View className="absolute z-10 inset-0 bg-white/50 items-center justify-center">
                                    <View className="w-48 h-48 rounded-full border-4 border-indigo-100 items-center justify-center animate-pulse">
                                        <View className="w-32 h-32 rounded-full border-4 border-indigo-200 items-center justify-center">
                                            <ActivityIndicator size="large" color="#4F46E5" />
                                        </View>
                                    </View>
                                    <Text className="text-indigo-600 font-bold mt-4 animate-bounce">Scanning Grid...</Text>
                                </View>
                            )}

                            {/* Peers List */}
                            <Text className="text-xs font-bold text-gray-400 uppercase mb-2">Nearby Nodes</Text>
                            {/* Log */}
                            <ScrollView className="mb-4 h-24" contentContainerStyle={{ flexGrow: 1, justifyContent: 'flex-end' }}>
                                {syncLog.map((log, i) => (
                                    <Text key={i} className="text-xs text-gray-500 font-mono">[{new Date().toLocaleTimeString()}] {log}</Text>
                                ))}
                            </ScrollView>

                            {foundPeers.length === 0 && !isScanning ? (
                                <View className="flex-1 items-center justify-center opacity-50">
                                    <Ionicons name="planet-outline" size={48} color="#9CA3AF" />
                                    <Text className="text-gray-400 mt-2">No active peers detected</Text>
                                </View>
                            ) : (
                                foundPeers.map(peer => (
                                    <View key={peer.id} className="bg-white p-4 rounded-xl border border-indigo-100 shadow-sm mb-3 flex-row items-center justify-between">
                                        <View className="flex-row items-center">
                                            <View className="w-10 h-10 bg-indigo-50 rounded-full items-center justify-center mr-3">
                                                <Ionicons name="bluetooth" size={20} color="#4F46E5" />
                                            </View>
                                            <View>
                                                <Text className="text-gray-800 font-bold">{peer.name}</Text>
                                                <Text className="text-green-600 text-xs font-bold">● {peer.distance}</Text>
                                            </View>
                                        </View>

                                        {!isSyncing ? (
                                            <TouchableOpacity
                                                onPress={() => startSync(peer)}
                                                className="bg-indigo-600 px-4 py-2 rounded-lg"
                                            >
                                                <Text className="text-white font-bold text-xs">Auto-Sync</Text>
                                            </TouchableOpacity>
                                        ) : (
                                            <View className="w-24">
                                                <View className="h-2 bg-gray-200 rounded-full overflow-hidden">
                                                    <View
                                                        className="h-full bg-indigo-500"
                                                        style={{ width: `${syncProgress * 100}%` }}
                                                    />
                                                </View>
                                                <Text className="text-[10px] text-center text-indigo-500 mt-1">Transferring...</Text>
                                            </View>
                                        )}
                                    </View>
                                ))
                            )}

                        </View>

                        {/* Scan Button */}
                        {!isScanning && !isSyncing && (
                            <TouchableOpacity
                                className="bg-gray-900 w-full py-4 rounded-xl items-center mt-4 active:bg-gray-800"
                                onPress={startScan}
                            >
                                <Ionicons name="refresh" size={20} color="#fff" />
                                <Text className="text-white font-bold text-lg ml-2">Scan for Peers</Text>
                            </TouchableOpacity>
                        )}
                    </View>
                </View>
            </Modal>

            {/* Settings Modal */}
            <Modal visible={showSettings} transparent animationType="fade">
                <View className="flex-1 bg-black/60 justify-end">
                    <View className="bg-white rounded-t-3xl p-6 pb-10">
                        <View className="flex-row justify-between items-center mb-6">
                            <Text className="text-2xl font-bold text-gray-800">Audio Settings</Text>
                            <View className="flex-row items-center gap-3">
                                <TouchableOpacity
                                    className="px-3 py-1 bg-gray-100 rounded-lg"
                                    onPress={() => setShowDevMenu(!showDevMenu)}
                                >
                                    <Text className="text-gray-500 text-sm font-medium">Dev</Text>
                                </TouchableOpacity>
                                <TouchableOpacity onPress={() => setShowSettings(false)}>
                                    <Ionicons name="close" size={28} color="#6B7280" />
                                </TouchableOpacity>
                            </View>
                        </View>

                        {/* Dev Menu */}
                        {showDevMenu && (
                            <View className="bg-gray-50 p-4 rounded-xl mb-4 border border-gray-200">
                                <Text className="text-sm font-bold text-gray-500 uppercase mb-3">Developer Options</Text>

                                <View className="bg-white p-3 rounded-lg border border-gray-100 flex-row items-center justify-between mb-3">
                                    <View className="flex-row items-center">
                                        <Ionicons name="flask-outline" size={20} color="#F59E0B" />
                                        <Text className="text-gray-700 font-medium ml-2">Testing / Demo Mode</Text>
                                    </View>
                                    <Switch
                                        trackColor={{ false: '#D1D5DB', true: '#F59E0B' }}
                                        thumbColor={'#FFFFFF'}
                                        ios_backgroundColor="#D1D5DB"
                                        onValueChange={setIsTestingMode}
                                        value={isTestingMode}
                                    />
                                </View>

                                <TouchableOpacity
                                    className="bg-red-100 p-3 rounded-lg flex-row items-center"
                                    onPress={async () => {
                                        await resetDatabase();
                                        setShowDevMenu(false);
                                        setShowSettings(false);
                                    }}
                                >
                                    <Ionicons name="trash-outline" size={20} color="#DC2626" />
                                    <Text className="text-red-600 font-medium ml-2">Reset Database (Dev Only)</Text>
                                </TouchableOpacity>
                            </View>
                        )}

                        <View className="mb-6">
                            <Text className="text-sm font-bold text-gray-500 uppercase tracking-wider mb-3">
                                Microphone Level
                            </Text>
                            <View className="h-8 bg-gray-100 rounded-full overflow-hidden">
                                <Animated.View
                                    className="h-full bg-red-500 rounded-full"
                                    style={{ width: levelWidth }}
                                />
                            </View>
                            <Text className="text-gray-400 text-sm mt-2">Speak to test your microphone</Text>
                        </View>

                        <Text className="text-sm font-bold text-gray-500 uppercase tracking-wider mb-3">
                            Audio Input
                        </Text>
                        <View className="flex-row items-center p-4 rounded-xl border-2 border-red-500 bg-red-50">
                            <View className="h-10 w-10 rounded-full items-center justify-center mr-4 bg-red-100">
                                <Ionicons name="mic" size={20} color="#DC2626" />
                            </View>
                            <View className="flex-1">
                                <Text className="text-base font-medium text-red-600">System Default</Text>
                                <Text className="text-xs text-gray-500 mt-1">Uses your device's active microphone</Text>
                            </View>
                            <Ionicons name="checkmark-circle" size={24} color="#DC2626" />
                        </View>

                        <TouchableOpacity
                            className="bg-red-600 py-4 rounded-xl items-center mt-6"
                            onPress={() => setShowSettings(false)}
                        >
                            <Text className="text-white font-bold text-lg">Done</Text>
                        </TouchableOpacity>

                        <TouchableOpacity
                            className="bg-red-100 py-4 rounded-xl items-center mt-3 flex-row justify-center"
                            onPress={async () => {
                                try {
                                    await AsyncStorage.removeItem('userToken');
                                } catch (e) { }
                                setShowSettings(false);
                                navigation.reset({ index: 0, routes: [{ name: 'Welcome' }] });
                            }}
                        >
                            <Ionicons name={isOffline ? "log-in-outline" : "log-out-outline"} size={20} color="#DC2626" />
                            <Text className="text-red-600 font-bold text-lg ml-2">{isOffline ? 'Log In' : 'Logout'}</Text>
                        </TouchableOpacity>
                    </View>
                </View>
            </Modal>
        </View>
    );
}
