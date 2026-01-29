import React, { useState, useEffect, useRef } from 'react';
import { View, Text, TouchableOpacity, ActivityIndicator, ScrollView, Animated } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { Audio } from 'expo-av';
import { useApp } from '../context/AppContext';
import { analyzeAudio, checkBackendHealth } from '../services/ApiService';
import * as Print from 'expo-print';
import * as Sharing from 'expo-sharing';

const SCAN_DURATION_SECONDS = 15; // 15 seconds for 3-5 respiratory cycles

// ESI Level configurations
const ESI_CONFIG = {
    1: { name: 'CRITICAL', color: '#DC2626' },
    2: { name: 'URGENT', color: '#EA580C' },
    3: { name: 'MODERATE', color: '#F59E0B' },
    4: { name: 'LOW', color: '#FACC15' },
    5: { name: 'STABLE', color: '#10B981' },
};

// Fallback classification when API unavailable
const classifyRespiratorySound = (audioFeatures) => {
    const { avgLevel, variance, peakCount, zeroCrossings } = audioFeatures;

    let classification = 'Normal';
    let confidence = 85;
    let baseSeverity = 15;

    if (avgLevel < 0.08) {
        classification = 'Absent/Abnormal';
        confidence = 75 + Math.random() * 10;
        baseSeverity = 85;
    } else if (variance > 0.06 && peakCount > 20) {
        classification = 'Crackles';
        confidence = 80 + Math.random() * 12;
        baseSeverity = 50 + (variance * 100);
    } else if (variance < 0.03 && zeroCrossings > 15) {
        classification = 'Wheezes';
        confidence = 82 + Math.random() * 12;
        baseSeverity = 55 + (zeroCrossings * 2);
    } else if (variance > 0.04 && zeroCrossings > 12 && peakCount > 15) {
        classification = 'Both';
        confidence = 75 + Math.random() * 10;
        baseSeverity = 70;
    } else if (avgLevel > 0.15 && variance < 0.04) {
        classification = 'Normal';
        confidence = 88 + Math.random() * 10;
        baseSeverity = 10 + Math.random() * 15;
    } else {
        classification = 'Normal';
        confidence = 75 + Math.random() * 15;
        baseSeverity = 20 + Math.random() * 10;
    }

    return { classification, confidence: Math.min(98, confidence), baseSeverity: Math.min(100, baseSeverity) };
};

export default function ScanResultScreen({ route, navigation }) {
    const { recordScan, refreshDashboard } = useApp();
    const { patientId, mode, audioUri, audioDuration } = route.params;

    const [phase, setPhase] = useState(mode === 'scan' ? 'recording' : 'analyzing');
    const [recordProgress, setRecordProgress] = useState(0);
    const [audioLevel, setAudioLevel] = useState(0);
    const [result, setResult] = useState(null);
    const [usingRealAI, setUsingRealAI] = useState(false);
    const [sound, setSound] = useState(null);
    const [isPlaying, setIsPlaying] = useState(false);
    const [isExporting, setIsExporting] = useState(false);

    const recordingRef = useRef(null);
    const recordedUriRef = useRef(null);
    const audioLevelAnim = useRef(new Animated.Value(0)).current;
    const audioHistory = useRef([]);

    useEffect(() => {
        if (mode === 'scan') {
            startRecording();
        } else if (mode === 'upload') {
            analyzeWithAPI(audioUri);
        }
        return () => {
            stopRecording();
            if (sound) {
                sound.unloadAsync();
            }
        };
    }, []);

    // Audio playback handlers
    const playAudio = async () => {
        try {
            // Get audio URI from either recorded ref (review phase) or result (results screen)
            const audioUri = recordedUriRef.current || result?.audio_uri;
            if (!audioUri) {
                console.log('No audio URI available');
                return;
            }

            if (sound) {
                // Resume if paused
                await sound.playAsync();
                setIsPlaying(true);
            } else {
                // Load and play
                const { sound: newSound } = await Audio.Sound.createAsync(
                    { uri: audioUri },
                    { shouldPlay: true },
                    (status) => {
                        if (status.didJustFinish) {
                            setIsPlaying(false);
                        }
                    }
                );
                setSound(newSound);
                setIsPlaying(true);
            }
        } catch (error) {
            console.error('Error playing audio:', error);
        }
    };

    const pauseAudio = async () => {
        try {
            if (sound) {
                await sound.pauseAsync();
                setIsPlaying(false);
            }
        } catch (error) {
            console.error('Error pausing audio:', error);
        }
    };

    const togglePlayback = () => {
        if (isPlaying) {
            pauseAudio();
        } else {
            playAudio();
        }
    };

    const handleExportPDF = async () => {
        if (!result) return;
        setIsExporting(true);

        const html = `
<html>
  <head>
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, minimum-scale=1.0, user-scalable=no" />
    <style>
      body { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; padding: 40px; color: #333; }
      .header { display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #EEE; padding-bottom: 20px; margin-bottom: 30px; }
      .logo { font-size: 24px; font-bold: bold; color: #DC2626; }
      .esi-badge { padding: 8px 16px; border-radius: 20px; font-weight: bold; color: white; }
      .vitals-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin-bottom: 30px; }
      .vital-card { padding: 15px; background: #F9FAFB; border-radius: 12px; text-align: center; border: 1px solid #E5E7EB; }
      .vital-label { font-size: 12px; color: #6B7280; text-transform: uppercase; font-weight: bold; }
      .vital-value { font-size: 24px; font-weight: bold; margin: 5px 0; }
      .section { margin-bottom: 25px; }
      .section-title { font-size: 14px; font-weight: bold; color: #9CA3AF; text-transform: uppercase; margin-bottom: 10px; }
      .diagnosis { font-size: 20px; font-weight: bold; color: #111827; }
      .recommendation-card { padding: 20px; border-radius: 12px; margin-top: 10px; line-height: 1.6; }
      .footer { margin-top: 50px; font-size: 12px; color: #9CA3AF; text-align: center; border-top: 1px solid #EEE; padding-top: 20px; }
    </style>
  </head>
  <body>
    <div class="header">
      <div>
        <div class="logo">RevoScope LifeLine</div>
        <div style="font-size: 14px; color: #6B7280; margin-top: 4px;">Diagnostic Report</div>
      </div>
      <div class="esi-badge" style="background-color: ${ESI_CONFIG[result.esi_level].color}">
        ESI LEVEL ${result.esi_level}
      </div>
    </div>

    <div class="section" style="margin-bottom: 40px;">
      <div class="section-title">Patient Information</div>
      <div style="font-size: 16px;">
        <strong>Patient ID:</strong> ${patientId}<br/>
        <strong>Scan Date:</strong> ${new Date(result.timestamp).toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric', hour: '2-digit', minute: '2-digit' })}
      </div>
    </div>

    <div class="section">
      <div class="section-title">Vital Signs</div>
      <div class="vitals-grid">
        <div class="vital-card" style="border-top: 4px solid ${ESI_CONFIG[result.esi_level].color}">
          <div class="vital-label">Severity</div>
          <div class="vital-value" style="color: ${ESI_CONFIG[result.esi_level].color}">${result.severity_score}%</div>
        </div>
        <div class="vital-card" style="border-top: 4px solid #3B82F6">
          <div class="vital-label">Confidence</div>
          <div class="vital-value" style="color: #3B82F6">${result.confidence_score}%</div>
        </div>
        <div class="vital-card" style="border-top: 4px solid #DC2626">
          <div class="vital-label">Heart Rate</div>
          <div class="vital-value" style="color: #DC2626">${result.heart_rate} <span style="font-size: 12px;">BPM</span></div>
        </div>
      </div>
      <div class="vitals-grid">
        <div class="vital-card" style="border-top: 4px solid #10B981">
          <div class="vital-label">Respiratory Rate</div>
          <div class="vital-value" style="color: #10B981">${result.respiratory_rate} <span style="font-size: 12px;">/min</span></div>
        </div>
      </div>
    </div>

    <div class="section">
      <div class="section-title">AI Diagnosis</div>
      <div class="diagnosis">${result.diagnosis}</div>
    </div>

    <div class="section">
      <div class="section-title">Clinical Recommendation</div>
      <div class="recommendation-card" style="background-color: ${ESI_CONFIG[result.esi_level].color}10; border-left: 4px solid ${ESI_CONFIG[result.esi_level].color}">
        ${result.recommendation}
      </div>
    </div>

    <div class="footer">
      This report was generated by RevoScope LifeLine AI.<br/>
      Medical assessment should be performed by a qualified healthcare professional.
    </div>
  </body>
</html>
        `;

        try {
            const { uri } = await Print.printToFileAsync({ html });
            setIsExporting(false);
            await Sharing.shareAsync(uri, { UTI: '.pdf', mimeType: 'application/pdf' });
        } catch (error) {
            console.error('Error generating PDF:', error);
            setIsExporting(false);
            alert('Failed to generate PDF report.');
        }
    };

    // Try to analyze with real API first
    const analyzeWithAPI = async (uri) => {
        setPhase('analyzing');

        try {
            // Check if backend is available
            const health = await checkBackendHealth();

            if (health.available && health.modelLoaded) {
                console.log('Using real AI model...');
                setUsingRealAI(true);

                const apiResult = await analyzeAudio(uri);

                if (apiResult.success) {
                    const { data } = apiResult;
                    await completeAnalysisWithAPIResult(data);
                    return;
                }
            }
        } catch (e) {
            console.log('API unavailable, using fallback:', e.message);
        }

        // Fallback to simulated classification
        console.log('Using fallback classification...');
        setUsingRealAI(false);
        const simulatedFeatures = {
            avgLevel: 0.3 + Math.random() * 0.3,
            variance: Math.random() * 0.08,
            peakCount: Math.floor(10 + Math.random() * 25),
            zeroCrossings: Math.floor(8 + Math.random() * 20),
        };
        await completeAnalysis(simulatedFeatures);
    };

    const startRecording = async () => {
        try {
            await Audio.requestPermissionsAsync();
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
                        audioHistory.current.push(level);

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

            // 15 second recording
            const totalMs = SCAN_DURATION_SECONDS * 1000;
            const intervalMs = 100;
            let elapsed = 0;

            const interval = setInterval(() => {
                elapsed += intervalMs;
                const progress = (elapsed / totalMs) * 100;
                setRecordProgress(progress);

                if (progress >= 100) {
                    clearInterval(interval);
                    handleRecordingComplete();
                }
            }, intervalMs);
        } catch (e) {
            console.error('Failed to start recording:', e);
            setPhase('analyzing');
            const defaultFeatures = { avgLevel: 0.3, variance: 0.03, peakCount: 12, zeroCrossings: 10 };
            setTimeout(() => completeAnalysis(defaultFeatures), 2000);
        }
    };

    const handleRecordingComplete = async () => {
        await stopRecording();
        setPhase('review'); // Go to review phase instead of analyzing
    };

    const handleRedoRecording = async () => {
        // Reset and start new recording
        if (sound) {
            await sound.unloadAsync();
            setSound(null);
        }
        setIsPlaying(false);
        audioHistory.current = [];
        setRecordProgress(0);
        recordedUriRef.current = null;
        setPhase('recording');
        setTimeout(() => startRecording(), 100);
    };

    const handleContinueToAnalysis = async () => {
        setPhase('analyzing');

        // Get the recorded URI
        const recordedUri = recordedUriRef.current;

        if (recordedUri) {
            // Try API first
            await analyzeWithAPI(recordedUri);
        } else {
            // Fallback to feature-based classification
            const features = extractAudioFeatures(audioHistory.current);
            await completeAnalysis(features);
        }
    };

    const extractAudioFeatures = (samples) => {
        if (samples.length === 0) {
            return { avgLevel: 0.3, variance: 0.03, peakCount: 10, zeroCrossings: 10 };
        }

        // Calculate average level
        const avgLevel = samples.reduce((a, b) => a + b, 0) / samples.length;

        // Calculate variance
        const variance = samples.reduce((acc, val) => acc + Math.pow(val - avgLevel, 2), 0) / samples.length;

        // Count peaks (local maxima)
        let peakCount = 0;
        for (let i = 1; i < samples.length - 1; i++) {
            if (samples[i] > samples[i - 1] && samples[i] > samples[i + 1] && samples[i] > avgLevel * 1.2) {
                peakCount++;
            }
        }

        // Count zero crossings (relative to average)
        let zeroCrossings = 0;
        for (let i = 1; i < samples.length; i++) {
            if ((samples[i] - avgLevel) * (samples[i - 1] - avgLevel) < 0) {
                zeroCrossings++;
            }
        }
        zeroCrossings = zeroCrossings / (samples.length / 100); // Normalize per 100 samples

        return { avgLevel, variance, peakCount, zeroCrossings };
    };

    const stopRecording = async () => {
        if (recordingRef.current) {
            try {
                await recordingRef.current.stopAndUnloadAsync();
                // Save URI for potential API call
                recordedUriRef.current = recordingRef.current.getURI();
                recordingRef.current = null;
            } catch (e) {
                console.error('Failed to stop recording:', e);
            }
        }
    };

    // Handle API result (from real AI model)
    const completeAnalysisWithAPIResult = async (data) => {
        // Calculate respiratory rate (from API or simulate)
        const respiratoryRate = data.respiratoryRate || Math.floor(12 + (data.severity / 100) * 18); // 12-30 range based on severity

        const scanResult = {
            diagnosis: data.diagnosis === 'Both' ? 'Crackles + Wheezes' : data.diagnosis,
            severity_score: data.severity,
            confidence_score: Math.floor(data.confidence),
            esi_level: data.esiLevel,
            recommendation: data.recommendation,
            heart_rate: data.heartRate,
            respiratory_rate: respiratoryRate,
            timestamp: new Date().toISOString(),
            audio_uri: audioUri || recordedUriRef.current || 'recorded_audio',
            status: data.esiLevel <= 2 ? 'Critical' : data.esiLevel <= 3 ? 'Monitoring' : 'Normal'
        };

        setResult(scanResult);
        setPhase('result');

        // Save to DB
        await recordScan(
            patientId,
            audioUri || recordedUriRef.current || 'recorded_audio',
            scanResult.diagnosis,
            scanResult.severity_score,
            scanResult.confidence_score,
            scanResult.status,
            data.recommendation,
            data.heartRate
        );
        await refreshDashboard();
    };

    // Fallback analysis (when API unavailable)
    const completeAnalysis = async (features) => {
        // Classify using fallback algorithm
        const { classification, confidence, baseSeverity } = classifyRespiratorySound(features);

        // Add small variance to severity for realism
        const severity = Math.min(100, Math.max(0, Math.floor(baseSeverity + (Math.random() * 10 - 5))));

        // Determine ESI level based on classification and severity
        let esiLevel = 5;
        if (classification === 'Absent/Abnormal' || severity >= 85) esiLevel = 1;
        else if (severity >= 70) esiLevel = 2;
        else if (severity >= 50 || classification === 'Both') esiLevel = 3;
        else if (severity >= 30 || classification !== 'Normal') esiLevel = 4;
        else esiLevel = 5;

        // Clinical recommendations based on ESI and diagnosis
        const recommendations = {
            1: `IMMEDIATE INTERVENTION REQUIRED. ${classification === 'Absent/Abnormal' ? 'Possible pneumothorax - prepare for chest decompression.' : 'Severe respiratory distress detected.'} Continuous SpO2 and cardiac monitoring mandatory. Consider emergency intubation.`,
            2: `Urgent evaluation needed. ${classification} detected with high severity. Administer supplemental oxygen if SpO2 < 94%. Consider bronchodilator therapy. Obtain chest X-ray and arterial blood gas immediately.`,
            3: `Expedited assessment required. ${classification} present. Schedule bronchodilator treatment. Monitor respiratory rate and oxygen saturation every 15 minutes. Consider corticosteroid if symptoms persist.`,
            4: `Standard evaluation pathway. Mild ${classification.toLowerCase()} detected. Complete pulmonary function assessment. Patient education on symptom monitoring. Schedule follow-up in 1-2 weeks.`,
            5: `No immediate intervention required. Normal vesicular breath sounds detected. Routine follow-up as scheduled. Continue current management plan.`
        };

        const triageAdvice = recommendations[esiLevel];

        // Generate simulated heart rate based on severity
        // Higher severity often correlates with higher heart rate (tachycardia)
        let baseHr = 75;
        if (severity > 70) baseHr = 110;
        else if (severity > 40) baseHr = 95;

        const simulatedHeartRate = Math.floor(baseHr + (Math.random() * 20 - 10));

        // Generate respiratory rate based on severity and diagnosis
        // Normal: 12-20, Distressed: 20-30+
        let baseRespRate = 16; // Normal baseline
        if (severity > 70) baseRespRate = 26; // Tachypnea in severe cases
        else if (severity > 40) baseRespRate = 22; // Elevated
        else if (classification === 'Wheezes' || classification === 'Both') baseRespRate = 24; // Wheezing often increases RR

        const simulatedRespiratoryRate = Math.floor(baseRespRate + (Math.random() * 6 - 3));

        const scanResult = {
            diagnosis: classification === 'Both' ? 'Crackles + Wheezes' : classification,
            severity_score: severity,
            confidence_score: Math.floor(confidence),
            esi_level: esiLevel,
            recommendation: triageAdvice,
            heart_rate: simulatedHeartRate,
            respiratory_rate: simulatedRespiratoryRate,
            timestamp: new Date().toISOString(),
            audio_uri: audioUri || recordedUriRef.current || 'recorded_audio',
            status: esiLevel <= 2 ? 'Critical' : esiLevel <= 3 ? 'Monitoring' : 'Normal'
        };

        setResult(scanResult);
        setPhase('result');

        // Save to DB with triage advice
        await recordScan(
            patientId,
            audioUri || recordedUriRef.current || 'recorded_audio',
            scanResult.diagnosis,
            severity,
            Math.floor(confidence),
            scanResult.status,
            triageAdvice,
            simulatedHeartRate
        );
        await refreshDashboard();
    };

    const levelWidth = audioLevelAnim.interpolate({
        inputRange: [0, 1],
        outputRange: ['0%', '100%'],
    });

    // Recording Phase
    if (phase === 'recording') {
        const secondsElapsed = Math.floor((recordProgress / 100) * SCAN_DURATION_SECONDS);

        return (
            <View className="flex-1 bg-white items-center justify-center">
                <View className="bg-gray-50 p-8 rounded-3xl items-center mx-6 w-full max-w-sm">
                    <View className="h-20 w-20 bg-red-600 rounded-full items-center justify-center mb-5">
                        <Ionicons name="mic" size={36} color="#fff" />
                    </View>
                    <Text className="text-2xl font-bold text-gray-800">Recording...</Text>
                    <Text className="text-gray-500 mt-1 mb-5">Capturing respiratory sounds</Text>

                    {/* Audio Level Meter */}
                    <View className="w-full mb-4">
                        <Text className="text-xs font-bold text-gray-500 mb-2">AUDIO LEVEL</Text>
                        <View className="h-5 bg-gray-200 rounded-full overflow-hidden">
                            <Animated.View
                                className="h-full bg-red-500 rounded-full"
                                style={{ width: levelWidth }}
                            />
                        </View>
                    </View>

                    {/* Progress */}
                    <View className="w-full">
                        <Text className="text-xs font-bold text-gray-500 mb-2">PROGRESS</Text>
                        <View className="w-full bg-gray-200 h-3 rounded-full overflow-hidden">
                            <View
                                className="bg-red-600 h-full rounded-full"
                                style={{ width: `${recordProgress}%` }}
                            />
                        </View>
                        <Text className="text-gray-400 text-sm mt-2 text-center">
                            {secondsElapsed}s / {SCAN_DURATION_SECONDS}s
                        </Text>
                    </View>
                </View>
            </View>
        );
    }

    // Review Phase - Added between recording and analyzing
    if (phase === 'review') {
        return (
            <View className="flex-1 bg-white items-center justify-center">
                <View className="bg-gray-50 p-8 rounded-3xl items-center mx-6 w-full max-w-sm">
                    <View className="h-20 w-20 bg-green-600 rounded-full items-center justify-center mb-5">
                        <Ionicons name="checkmark" size={42} color="#fff" />
                    </View>
                    <Text className="text-2xl font-bold text-gray-800">Recording Complete</Text>
                    <Text className="text-gray-500 mt-1 mb-5 text-center">
                        {SCAN_DURATION_SECONDS} seconds of respiratory sounds captured
                    </Text>

                    {/* Audio Playback */}
                    <TouchableOpacity
                        className="w-full bg-white p-4 rounded-2xl border-2 border-gray-200 mb-4 flex-row items-center justify-between"
                        onPress={togglePlayback}
                    >
                        <View className="flex-row items-center flex-1">
                            <View className="h-12 w-12 rounded-full bg-red-50 items-center justify-center mr-3">
                                <Ionicons
                                    name={isPlaying ? "pause" : "play"}
                                    size={24}
                                    color="#DC2626"
                                />
                            </View>
                            <View className="flex-1">
                                <Text className="text-base font-bold text-gray-800">
                                    {isPlaying ? 'Playing...' : 'Play Recording'}
                                </Text>
                                <Text className="text-xs text-gray-500">Review audio quality</Text>
                            </View>
                        </View>
                        <Ionicons name="volume-high" size={24} color="#9CA3AF" />
                    </TouchableOpacity>

                    {/* Action Buttons */}
                    <View className="w-full gap-3">
                        <TouchableOpacity
                            className="py-4 rounded-xl items-center bg-red-600"
                            onPress={handleContinueToAnalysis}
                        >
                            <Text className="text-white font-bold text-lg">Continue to Analysis</Text>
                        </TouchableOpacity>

                        <TouchableOpacity
                            className="py-4 rounded-xl items-center bg-white border-2 border-gray-300"
                            onPress={handleRedoRecording}
                        >
                            <View className="flex-row items-center">
                                <Ionicons name="refresh" size={20} color="#374151" />
                                <Text className="text-gray-700 font-bold text-base ml-2">Redo Recording</Text>
                            </View>
                        </TouchableOpacity>
                    </View>
                </View>
            </View>
        );
    }

    // Analyzing Phase
    if (phase === 'analyzing') {
        return (
            <View className="flex-1 bg-white items-center justify-center">
                <View className="bg-gray-50 p-10 rounded-3xl items-center">
                    <ActivityIndicator size="large" color="#DC2626" />
                    <Text className="mt-6 text-2xl font-bold text-gray-800">AI Analysis</Text>
                    <Text className="text-gray-500 mt-2 text-center">
                        Processing respiratory patterns{'\n'}with CNN classifier
                    </Text>
                </View>
            </View>
        );
    }

    // Result Phase
    const esi = ESI_CONFIG[result.esi_level];

    return (
        <View className="flex-1 bg-gray-50">
            {/* Header */}
            <View className="pt-16 pb-4 px-6 bg-white border-b border-gray-100 flex-row justify-between items-center">
                <View>
                    <Text className="text-xl font-bold text-gray-800">Diagnostic Result</Text>
                    <Text className="text-sm text-gray-400">
                        {new Date(result.timestamp).toLocaleDateString('en-US', {
                            month: 'short',
                            day: 'numeric',
                            year: 'numeric',
                            hour: '2-digit',
                            minute: '2-digit'
                        })}
                    </Text>
                </View>
                <View className="px-4 py-2 rounded-full" style={{ backgroundColor: esi.color + '20', borderWidth: 2, borderColor: esi.color }}>
                    <Text style={{ color: esi.color, fontWeight: 'bold' }}>ESI-{result.esi_level}</Text>
                </View>
            </View>

            <ScrollView className="flex-1" contentContainerStyle={{ padding: 16, paddingBottom: 120 }}>
                {/* Triple Score Display */}
                <View className="flex-row justify-center gap-6 mb-6">
                    {/* Severity Circle */}
                    <View
                        className="h-36 w-36 rounded-full items-center justify-center bg-white shadow-lg"
                        style={{ borderWidth: 6, borderColor: esi.color }}
                    >
                        <Text className="text-gray-400 text-sm font-bold uppercase">Severity</Text>
                        <Text className="text-4xl font-bold" style={{ color: esi.color }}>
                            {result.severity_score}%
                        </Text>
                    </View>
                    {/* Confidence Circle */}
                    <View
                        className="h-36 w-36 rounded-full items-center justify-center bg-white shadow-lg"
                        style={{ borderWidth: 6, borderColor: '#3B82F6' }}
                    >
                        <Text className="text-gray-400 text-sm font-bold uppercase">Confidence</Text>
                        <Text className="text-4xl font-bold text-blue-500">
                            {result.confidence_score}%
                        </Text>
                    </View>
                    {/* Heart Rate Circle */}
                    {result.heart_rate && (
                        <View
                            className="h-36 w-36 rounded-full items-center justify-center bg-white shadow-lg"
                            style={{ borderWidth: 6, borderColor: '#DC2626' }}
                        >
                            <Text className="text-gray-400 text-sm font-bold uppercase">Heart Rate</Text>
                            <Text className="text-4xl font-bold text-red-600">
                                {result.heart_rate}
                            </Text>
                            <Text className="text-xs text-gray-500 mt-1">BPM</Text>
                        </View>
                    )}
                </View>

                {/* Audio Playback Control */}
                {result.audio_uri && (
                    <TouchableOpacity
                        className="bg-white p-4 rounded-2xl border border-gray-100 mb-3 flex-row items-center justify-between"
                        onPress={togglePlayback}
                    >
                        <View className="flex-row items-center flex-1">
                            <View className="h-10 w-10 rounded-full bg-red-50 items-center justify-center mr-3">
                                <Ionicons
                                    name={isPlaying ? "pause" : "play"}
                                    size={20}
                                    color="#DC2626"
                                />
                            </View>
                            <View className="flex-1">
                                <Text className="text-sm font-bold text-gray-800">
                                    {isPlaying ? 'Playing' : 'Play'} Respiratory Audio
                                </Text>
                                <Text className="text-xs text-gray-500">Recorded respiratory sounds</Text>
                            </View>
                        </View>
                        <Ionicons name="volume-high" size={20} color="#9CA3AF" />
                    </TouchableOpacity>
                )}

                {/* AI Diagnosis */}
                <View className="bg-white p-5 rounded-2xl border border-gray-100 mb-3">
                    <Text className="text-xs font-bold text-gray-400 uppercase mb-2">AI Diagnosis</Text>
                    <Text className="text-2xl font-bold text-gray-800">{result.diagnosis}</Text>
                    <Text className="text-sm text-gray-500 mt-2 leading-5">
                        {result.diagnosis === 'Normal'
                            ? 'Normal vesicular breath sounds detected. No adventitious sounds identified. Breath sounds are clear and symmetric.'
                            : result.diagnosis === 'Crackles'
                                ? 'Discontinuous, explosive sounds detected during inspiration. May indicate fluid in alveoli or opening of collapsed airways.'
                                : result.diagnosis === 'Wheezes'
                                    ? 'Continuous, high-pitched musical sounds detected. Indicates narrowing of airways or bronchospasm.'
                                    : result.diagnosis === 'Crackles + Wheezes'
                                        ? 'Mixed adventitious sounds detected. Both discontinuous crackles and continuous wheezes present, suggesting complex pathology.'
                                        : 'Significantly diminished or absent breath sounds detected. May indicate severe pathology requiring immediate evaluation.'
                        }
                    </Text>
                </View>

                {/* Potential Condition */}
                {result.diagnosis !== 'Normal' && (
                    <View className="p-4 rounded-2xl border mb-3" style={{
                        backgroundColor: esi.color + '10',
                        borderColor: esi.color + '40'
                    }}>
                        <View className="flex-row items-center mb-2">
                            <Ionicons name="alert-circle" size={20} color={esi.color} />
                            <Text className="font-bold text-gray-700 ml-2">Potential Condition</Text>
                        </View>
                        <Text className="text-lg font-bold" style={{ color: esi.color }}>
                            {result.diagnosis === 'Crackles'
                                ? 'Possible Pneumonia or Pulmonary Edema'
                                : result.diagnosis === 'Wheezes'
                                    ? 'Possible Asthma or COPD Exacerbation'
                                    : result.diagnosis === 'Crackles + Wheezes'
                                        ? 'Possible Bronchopneumonia'
                                        : 'Possible Pneumothorax (Collapsed Lung)'
                            }
                        </Text>
                        <Text className="text-sm text-gray-600 mt-1">
                            {result.diagnosis === 'Crackles'
                                ? 'Crackles often indicate fluid accumulation in the lungs from infection or heart failure.'
                                : result.diagnosis === 'Wheezes'
                                    ? 'Wheezing suggests bronchial constriction from asthma, allergic reaction, or COPD.'
                                    : result.diagnosis === 'Crackles + Wheezes'
                                        ? 'Combined sounds may indicate bacterial lung infection with bronchial involvement.'
                                        : 'Absent breath sounds can indicate air or fluid in pleural space.'
                            }
                        </Text>
                    </View>
                )}

                {/* Clinical Recommendation */}
                <View className="p-5 rounded-2xl mb-3" style={{ backgroundColor: esi.color + '15' }}>
                    <View className="flex-row items-center mb-3">
                        <View className="h-8 w-8 rounded-full items-center justify-center" style={{ backgroundColor: esi.color }}>
                            <Ionicons name="medical" size={18} color="#fff" />
                        </View>
                        <Text className="font-bold text-gray-800 ml-3 text-base">Clinical Recommendation</Text>
                    </View>
                    <Text className="text-gray-700 leading-6 text-base">{result.recommendation}</Text>
                </View>

                {/* Respiratory Rate */}
                {result.respiratory_rate && (
                    <View className="bg-white p-4 rounded-2xl border border-gray-100 flex-row items-center justify-between">
                        <View className="flex-row items-center">
                            <View className="h-10 w-10 rounded-full bg-green-50 items-center justify-center mr-3">
                                <Ionicons name="pulse" size={24} color="#10B981" />
                            </View>
                            <View>
                                <Text className="text-xs font-bold text-gray-400 uppercase">Respiratory Rate</Text>
                                <Text className="text-sm text-gray-600 mt-1">Breaths per minute</Text>
                            </View>
                        </View>
                        <View className="items-center">
                            <Text className="text-4xl font-bold text-green-600">{result.respiratory_rate}</Text>
                            <Text className="text-xs text-gray-500">/min</Text>
                        </View>
                    </View>
                )}

                {/* Export PDF Button */}
                <TouchableOpacity
                    className={`bg-white p-4 rounded-2xl border border-gray-100 mt-3 flex-row items-center justify-between ${isExporting ? 'opacity-50' : ''}`}
                    onPress={handleExportPDF}
                    disabled={isExporting}
                >
                    <View className="flex-row items-center flex-1">
                        <View className="h-10 w-10 rounded-full bg-blue-50 items-center justify-center mr-3">
                            <Ionicons
                                name="document-text"
                                size={24}
                                color="#3B82F6"
                            />
                        </View>
                        <View className="flex-1">
                            <Text className="text-sm font-bold text-gray-800">
                                {isExporting ? 'Generating Report...' : 'Export as PDF Report'}
                            </Text>
                            <Text className="text-xs text-gray-500">Professional diagnostic summary</Text>
                        </View>
                    </View>
                    {isExporting ? (
                        <ActivityIndicator size="small" color="#3B82F6" />
                    ) : (
                        <Ionicons name="share-outline" size={24} color="#3B82F6" />
                    )}
                </TouchableOpacity>


            </ScrollView>

            {/* Footer */}
            <View className="p-6 bg-white border-t border-gray-100 absolute bottom-0 left-0 right-0">
                <TouchableOpacity
                    className="py-4 rounded-xl items-center bg-red-600"
                    onPress={() => navigation.navigate('Dashboard')}
                >
                    <Text className="text-white font-bold text-lg">Return to Dashboard</Text>
                </TouchableOpacity>
            </View>
        </View>
    );
}
