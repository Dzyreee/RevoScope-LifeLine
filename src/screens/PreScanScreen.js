import React, { useState, useEffect, useRef } from 'react';
import { View, Text, TouchableOpacity, ActivityIndicator, Animated, Alert } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { Audio } from 'expo-av';
import * as DocumentPicker from 'expo-document-picker';

const MIN_AUDIO_DURATION = 15; // seconds

export default function PreScanScreen({ route, navigation }) {
    const { patientId, includeHeartRate, forcedResult } = route.params;
    const [isStarting, setIsStarting] = useState(false);
    const [audioLevel, setAudioLevel] = useState(0);
    const [micWorking, setMicWorking] = useState(false);
    const audioLevelAnim = useRef(new Animated.Value(0)).current;
    const recordingRef = useRef(null);

    useEffect(() => {
        startTestRecording();
        return () => stopTestRecording();
    }, []);

    const startTestRecording = async () => {
        try {
            const { status } = await Audio.requestPermissionsAsync();
            if (status !== 'granted') return;

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
                        setMicWorking(level > 0.05);

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
            console.error('Failed to start test recording:', e);
        }
    };

    const stopTestRecording = async () => {
        if (recordingRef.current) {
            try {
                await recordingRef.current.stopAndUnloadAsync();
                recordingRef.current = null;
            } catch (e) {
                console.error('Failed to stop test recording:', e);
            }
        }
    };

    const handleBeginScan = async () => {
        setIsStarting(true);
        await stopTestRecording();
        setTimeout(() => {
            navigation.replace('Result', { patientId, mode: 'scan', includeHeartRate, forcedResult });
        }, 300);
    };

    const handleUploadAudio = async () => {
        try {
            const result = await DocumentPicker.getDocumentAsync({
                type: 'audio/*',
                copyToCacheDirectory: true,
            });

            if (result.canceled || !result.assets?.[0]) {
                return;
            }

            const file = result.assets[0];

            // Check audio duration
            const sound = new Audio.Sound();
            await sound.loadAsync({ uri: file.uri });
            const status = await sound.getStatusAsync();
            await sound.unloadAsync();

            if (status.isLoaded && status.durationMillis) {
                const durationSeconds = status.durationMillis / 1000;

                if (durationSeconds < MIN_AUDIO_DURATION) {
                    Alert.alert(
                        'Audio Too Short',
                        `Audio file must be at least ${MIN_AUDIO_DURATION} seconds long for accurate analysis. Your file is ${durationSeconds.toFixed(1)} seconds.`,
                        [{ text: 'OK' }]
                    );
                    return;
                }

                // Valid audio, navigate to analysis
                await stopTestRecording();
                navigation.replace('Result', {
                    patientId,
                    mode: 'upload',
                    audioUri: file.uri,
                    audioDuration: durationSeconds,
                    includeHeartRate,
                    forcedResult
                });
            } else {
                Alert.alert('Error', 'Could not read audio file duration.');
            }
        } catch (e) {
            console.error('Upload error:', e);
            Alert.alert('Error', 'Failed to process audio file.');
        }
    };

    const levelWidth = audioLevelAnim.interpolate({
        inputRange: [0, 1],
        outputRange: ['0%', '100%'],
    });

    return (
        <View className="flex-1 bg-white">
            {/* Header */}
            <View className="pt-16 pb-4 px-6 bg-white border-b border-gray-100 flex-row items-center">
                <TouchableOpacity onPress={() => navigation.goBack()} className="mr-4">
                    <Ionicons name="arrow-back" size={24} color="#374151" />
                </TouchableOpacity>
                <Text className="text-xl font-bold text-gray-800">Prepare Scan</Text>
            </View>

            <View className="flex-1 items-center justify-center px-6">
                {/* Microphone Icon */}
                <View className="h-28 w-28 bg-red-50 rounded-full items-center justify-center mb-5">
                    <View className="h-20 w-20 bg-red-100 rounded-full items-center justify-center">
                        <Ionicons name="mic" size={42} color="#DC2626" />
                    </View>
                </View>

                <Text className="text-2xl font-bold text-gray-800 text-center mb-2">
                    Ready to Record
                </Text>
                <Text className="text-base text-gray-500 text-center mb-6">
                    Recording will capture {MIN_AUDIO_DURATION} seconds{'\n'}for 3-5 respiratory cycles.
                </Text>

                {/* PCG Heart Rate Tip */}
                {includeHeartRate && (
                    <View className="bg-blue-50 border border-blue-100 p-4 rounded-xl mb-6 w-full flex-row items-start">
                        <Ionicons name="information-circle" size={22} color="#3B82F6" style={{ marginTop: 2, marginRight: 10 }} />
                        <View className="flex-1">
                            <Text className="font-bold text-blue-900 mb-1">Heart Rate Analysis Enabled</Text>
                            <Text className="text-blue-800 text-sm leading-5">
                                Reviewing breathing sounds for heart palpitations. For best accuracy, place stethoscope firmly and minimize ambient noise.
                            </Text>
                        </View>
                    </View>
                )}

                {/* Live Audio Level */}
                <View className="w-full bg-gray-50 rounded-2xl p-4 border border-gray-100 mb-4">
                    <Text className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-2">
                        Microphone Level
                    </Text>
                    <View className="h-6 bg-gray-200 rounded-full overflow-hidden">
                        <Animated.View
                            className="h-full bg-red-500 rounded-full"
                            style={{ width: levelWidth }}
                        />
                    </View>
                    <Text className="text-gray-400 text-xs mt-1">
                        {micWorking ? '✓ Microphone is working' : 'Speak to test microphone'}
                    </Text>
                </View>

                {/* Checklist */}
                <View className="bg-gray-50 rounded-2xl p-4 w-full border border-gray-100 mb-4">
                    <View className="flex-row items-center mb-2">
                        <Ionicons
                            name={micWorking ? "checkmark-circle" : "ellipse-outline"}
                            size={20}
                            color={micWorking ? "#10B981" : "#9CA3AF"}
                        />
                        <Text className={`ml-3 text-sm ${micWorking ? 'text-gray-700' : 'text-gray-400'}`}>
                            Audio input detected
                        </Text>
                    </View>
                    <View className="flex-row items-center mb-2">
                        <Ionicons name="checkmark-circle" size={20} color="#10B981" />
                        <Text className="text-gray-700 ml-3 text-sm">Patient profile saved</Text>
                    </View>
                    <View className="flex-row items-center">
                        <Ionicons name="checkmark-circle" size={20} color="#10B981" />
                        <Text className="text-gray-700 ml-3 text-sm">System ready</Text>
                    </View>
                </View>
            </View>

            {/* Button Area */}
            <View className="p-6 bg-white border-t border-gray-100">
                {/* Upload Audio Button */}
                <TouchableOpacity
                    className="bg-gray-100 py-4 rounded-xl items-center flex-row justify-center mb-3"
                    onPress={handleUploadAudio}
                >
                    <Ionicons name="cloud-upload-outline" size={22} color="#374151" />
                    <Text className="text-gray-700 font-bold text-base ml-2">Upload Audio File</Text>
                </TouchableOpacity>

                {/* Begin Scan Button */}
                <TouchableOpacity
                    className="bg-red-600 py-5 rounded-xl items-center flex-row justify-center"
                    onPress={handleBeginScan}
                    disabled={isStarting}
                >
                    {isStarting ? (
                        <ActivityIndicator color="#fff" />
                    ) : (
                        <>
                            <Ionicons name="play" size={24} color="#fff" />
                            <Text className="text-white font-bold text-lg ml-2">Begin Scan</Text>
                        </>
                    )}
                </TouchableOpacity>
            </View>
        </View>
    );
}
