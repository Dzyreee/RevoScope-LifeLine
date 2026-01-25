import React, { useState, useEffect } from 'react';
import { View, Text, Modal, StyleSheet, TouchableOpacity, Button } from 'react-native';
import { useApp } from '../context/AppContext';
import { requestAudioPermissions, startAudioStream, stopAudioStream } from '../services/AudioService';
import AudioWaveform from './AudioWaveform';

export default function DeviceSetupModal({ visible, onClose }) {
    const { audioSource, setAudioSource } = useApp();
    const [metering, setMetering] = useState(-160);
    const [recording, setRecording] = useState(null);

    useEffect(() => {
        let activeRecording = null;
        if (visible) {
            (async () => {
                const hasPermission = await requestAudioPermissions();
                if (hasPermission) {
                    activeRecording = await startAudioStream((level) => {
                        setMetering(level);
                    });
                    setRecording(activeRecording);
                }
            })();
        } else {
            stopAudio();
        }

        return () => {
            stopAudio();
        };
    }, [visible]);

    const stopAudio = async () => {
        if (recording) {
            await stopAudioStream(recording);
            setRecording(null);
        }
    };

    const handleSourceSelect = (source) => {
        setAudioSource(source);
        // In a real app, this would toggle hardware inputs.
        // Here we just set the state which other components can read.
    };

    return (
        <Modal visible={visible} animationType="slide" transparent>
            <View style={styles.overlay}>
                <View style={styles.modalContent}>
                    <Text style={styles.title}>Audio Input Configuration</Text>
                    <Text style={styles.subtitle}>Select Source</Text>

                    <View style={styles.options}>
                        <TouchableOpacity
                            style={[styles.option, audioSource === 'internal' && styles.optionSelected]}
                            onPress={() => handleSourceSelect('internal')}
                        >
                            <Text style={[styles.optionText, audioSource === 'internal' && styles.optionTextSelected]}>
                                Internal Mic
                            </Text>
                        </TouchableOpacity>

                        <TouchableOpacity
                            style={[styles.option, audioSource === 'external' && styles.optionSelected]}
                            onPress={() => handleSourceSelect('external')}
                        >
                            <Text style={[styles.optionText, audioSource === 'external' && styles.optionTextSelected]}>
                                Digital Stethoscope (External)
                            </Text>
                        </TouchableOpacity>
                    </View>

                    <View style={styles.waveformContainer}>
                        <Text style={styles.waveformLabel}>Signal Check</Text>
                        <AudioWaveform metering={metering} />
                        <Text style={styles.statusText}>
                            {metering > -50 ? "Signal Detected" : "Low / No Signal"}
                        </Text>
                    </View>

                    <Button title="Confirm & Close" onPress={onClose} />
                </View>
            </View>
        </Modal>
    );
}

const styles = StyleSheet.create({
    overlay: {
        flex: 1,
        backgroundColor: 'rgba(0,0,0,0.5)',
        justifyContent: 'center',
        padding: 20,
    },
    modalContent: {
        backgroundColor: '#fff',
        borderRadius: 12,
        padding: 20,
    },
    title: {
        fontSize: 20,
        fontWeight: 'bold',
        marginBottom: 8,
        textAlign: 'center',
    },
    subtitle: {
        fontSize: 16,
        color: '#666',
        marginBottom: 16,
        textAlign: 'center',
    },
    options: {
        flexDirection: 'row',
        marginBottom: 24,
        justifyContent: 'center',
    },
    option: {
        paddingVertical: 10,
        paddingHorizontal: 16,
        borderWidth: 1,
        borderColor: '#ccc',
        marginHorizontal: 4,
        borderRadius: 8,
    },
    optionSelected: {
        backgroundColor: '#E3F2FD',
        borderColor: '#2196F3',
    },
    optionText: {
        color: '#333',
    },
    optionTextSelected: {
        color: '#2196F3',
        fontWeight: 'bold',
    },
    waveformContainer: {
        marginBottom: 24,
        alignItems: 'center',
    },
    waveformLabel: {
        marginBottom: 8,
        fontSize: 14,
        color: '#666',
    },
    statusText: {
        marginTop: 8,
        fontSize: 12,
        color: '#666',
    }
});
