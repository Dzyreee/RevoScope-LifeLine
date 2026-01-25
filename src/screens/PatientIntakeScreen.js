import React, { useState } from 'react';
import { View, Text, TextInput, StyleSheet, Button, ScrollView, Image, TouchableOpacity, Alert } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { useApp } from '../context/AppContext';

export default function PatientIntakeScreen({ navigation }) {
    const { addNewPatient } = useApp();

    const [name, setName] = useState('');
    const [age, setAge] = useState('');
    const [complaint, setComplaint] = useState('');
    const [photo, setPhoto] = useState(null);

    const takePhoto = async () => {
        const permissionResult = await ImagePicker.requestCameraPermissionsAsync();

        if (permissionResult.granted === false) {
            alert("You've refused to allow this app into your camera!");
            return;
        }

        const result = await ImagePicker.launchCameraAsync({
            mediaTypes: ImagePicker.MediaType.Images,
            allowsEditing: true,
            quality: 0.5,
            // base64: true, // if we wanted base64
        });

        if (!result.canceled) {
            setPhoto(result.assets[0].uri);
        }
    };

    const handleSave = async () => {
        if (!name || !complaint) {
            Alert.alert("Missing Info", "Please ensure Name and Chief Complaint are filled.");
            return;
        }

        try {
            await addNewPatient({
                name,
                estimated_age: age ? parseInt(age) : null,
                chief_complaint: complaint,
                photo_uri: photo,
                triage_category: 'P3' // Default to stable until scanned
            });
            // Navigate back or to Scan?
            // Prompt implies "Pre-Scan Saving... before 'Perform Scan' button becomes active".
            // We can navigate back to Dashboard or a PatientDetail screen.
            // For now, let's go back to Dashboard.
            navigation.goBack();
        } catch (e) {
            Alert.alert("Error", "Could not save patient.");
        }
    };

    return (
        <ScrollView style={styles.container}>
            <Text style={styles.header}>New Patient Entry</Text>

            <TouchableOpacity style={styles.photoButton} onPress={takePhoto}>
                {photo ? (
                    <Image source={{ uri: photo }} style={styles.photo} />
                ) : (
                    <View style={styles.photoPlaceholder}>
                        <Text style={styles.photoText}>Tap to Take Photo</Text>
                    </View>
                )}
            </TouchableOpacity>

            <Text style={styles.label}>Full Name / ID</Text>
            <TextInput
                style={styles.input}
                value={name}
                onChangeText={setName}
                placeholder="Unknown Subject"
            />

            <Text style={styles.label}>Estimated Age</Text>
            <TextInput
                style={styles.input}
                value={age}
                onChangeText={setAge}
                keyboardType="numeric"
                placeholder="e.g. 35"
            />

            <Text style={styles.label}>Chief Complaint / Injury</Text>
            <TextInput
                style={[styles.input, styles.textArea]}
                value={complaint}
                onChangeText={setComplaint}
                multiline
                numberOfLines={3}
                placeholder="Brief description of injury"
            />

            <View style={styles.buttonContainer}>
                <Button title="Save Profile (Enable Scan)" onPress={handleSave} />
            </View>
        </ScrollView>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        padding: 20,
        backgroundColor: '#fff',
    },
    header: {
        fontSize: 24,
        fontWeight: 'bold',
        marginBottom: 20,
        color: '#333',
    },
    photoButton: {
        alignSelf: 'center',
        marginBottom: 20,
    },
    photo: {
        width: 150,
        height: 150,
        borderRadius: 75,
    },
    photoPlaceholder: {
        width: 150,
        height: 150,
        borderRadius: 75,
        backgroundColor: '#e0e0e0',
        justifyContent: 'center',
        alignItems: 'center',
        borderWidth: 1,
        borderColor: '#ccc',
        borderStyle: 'dashed',
    },
    photoText: {
        color: '#757575',
    },
    label: {
        fontSize: 16,
        fontWeight: '600',
        marginBottom: 8,
        color: '#333',
    },
    input: {
        borderWidth: 1,
        borderColor: '#ccc',
        borderRadius: 8,
        padding: 12,
        fontSize: 16,
        marginBottom: 16,
        backgroundColor: '#fafafa',
    },
    textArea: {
        height: 80,
        textAlignVertical: 'top',
    },
    buttonContainer: {
        marginTop: 20,
        marginBottom: 40,
    },
});
