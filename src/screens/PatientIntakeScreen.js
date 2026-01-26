import React, { useState } from 'react';
import { View, Text, TextInput, TouchableOpacity, ScrollView, Alert, Image } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import * as ImagePicker from 'expo-image-picker';
import { useApp } from '../context/AppContext';

const MAX_NAME_LENGTH = 100;
const MAX_HISTORY_LENGTH = 1000;
const MAX_AGE = 150;

export default function PatientIntakeScreen({ route, navigation }) {
    const { createPatient, updatePatient } = useApp();

    const editMode = route.params?.editMode || false;
    const existingData = route.params?.patientData || null;

    const [fullName, setFullName] = useState(existingData?.full_name || '');
    const [age, setAge] = useState(existingData?.age?.toString() || '');
    const [sex, setSex] = useState(existingData?.sex || null);
    const [history, setHistory] = useState(existingData?.history || '');
    const [photo, setPhoto] = useState(existingData?.profile_image || null);
    const [heartRate, setHeartRate] = useState(existingData?.heart_rate?.toString() || '');

    // Validation states
    const isNameTooLong = fullName.length > MAX_NAME_LENGTH;
    const isHistoryTooLong = history.length > MAX_HISTORY_LENGTH;
    const isAgeInvalid = age && parseInt(age) > MAX_AGE;
    const historyCharsLeft = MAX_HISTORY_LENGTH - history.length;

    const handlePickPhoto = async () => {
        const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
        if (status !== 'granted') {
            Alert.alert('Permission Required', 'Camera roll access is needed.');
            return;
        }

        const result = await ImagePicker.launchImageLibraryAsync({
            mediaTypes: ImagePicker.MediaTypeOptions.Images,
            allowsEditing: true,
            aspect: [1, 1],
            quality: 0.8,
        });

        if (!result.canceled && result.assets?.[0]) {
            setPhoto(result.assets[0].uri);
        }
    };

    const handleTakePhoto = async () => {
        const { status } = await ImagePicker.requestCameraPermissionsAsync();
        if (status !== 'granted') {
            Alert.alert('Permission Required', 'Camera access is needed.');
            return;
        }

        const result = await ImagePicker.launchCameraAsync({
            allowsEditing: true,
            aspect: [1, 1],
            quality: 0.8,
        });

        if (!result.canceled && result.assets?.[0]) {
            setPhoto(result.assets[0].uri);
        }
    };

    const showPhotoOptions = () => {
        Alert.alert('Profile Photo', 'Choose an option', [
            { text: 'Take Photo', onPress: handleTakePhoto },
            { text: 'Choose from Library', onPress: handlePickPhoto },
            { text: 'Cancel', style: 'cancel' },
        ]);
    };

    const handleAgeChange = (text) => {
        const numericValue = text.replace(/[^0-9]/g, '');
        setAge(numericValue);
    };

    const handleSave = async () => {
        if (!fullName.trim()) {
            Alert.alert('Missing Info', 'Please enter the patient name.');
            return;
        }
        if (isNameTooLong) {
            Alert.alert('Invalid Name', `Name must be ${MAX_NAME_LENGTH} characters or less.`);
            return;
        }
        if (!age || parseInt(age) < 1) {
            Alert.alert('Missing Info', 'Please enter a valid age.');
            return;
        }
        if (isAgeInvalid) {
            Alert.alert('Invalid Age', `Age must be ${MAX_AGE} or less.`);
            return;
        }
        if (!sex) {
            Alert.alert('Missing Info', 'Please select sex.');
            return;
        }
        if (isHistoryTooLong) {
            Alert.alert('Invalid History', `Medical history must be ${MAX_HISTORY_LENGTH} characters or less.`);
            return;
        }

        try {
            const patientData = {
                full_name: fullName.trim(),
                age: parseInt(age),
                sex,
                history: history.trim(),
                profile_image: photo,
                heart_rate: heartRate ? parseInt(heartRate) : null,
            };

            if (editMode && existingData?.id) {
                await updatePatient(existingData.id, patientData);
                navigation.goBack();
            } else {
                const newId = await createPatient({
                    ...patientData,
                    severity_score: 0,
                    confidence_score: 0,
                    triage_advice: 'Pending Scan'
                });
                navigation.replace('PreScan', { patientId: newId });
            }
        } catch (e) {
            console.error(e);
            Alert.alert('Error', 'Could not save patient record.');
        }
    };

    return (
        <View className="flex-1 bg-white">
            {/* Header */}
            <View className="pt-16 pb-4 px-6 bg-white border-b border-gray-100 flex-row items-center">
                <TouchableOpacity onPress={() => navigation.goBack()} className="mr-4">
                    <Ionicons name="arrow-back" size={24} color="#374151" />
                </TouchableOpacity>
                <Text className="text-xl font-bold text-gray-800">
                    {editMode ? 'Edit Patient' : 'New Patient'}
                </Text>
            </View>

            <ScrollView className="flex-1 px-6 py-6" contentContainerStyle={{ paddingBottom: 120 }}>

                {/* Photo Upload */}
                <View className="items-center mb-8">
                    <TouchableOpacity
                        className="h-28 w-28 bg-gray-100 rounded-full items-center justify-center border-2 border-dashed border-gray-300 overflow-hidden"
                        onPress={showPhotoOptions}
                    >
                        {photo ? (
                            <Image source={{ uri: photo }} className="w-full h-full" />
                        ) : (
                            <View className="items-center">
                                <Ionicons name="camera" size={28} color="#9CA3AF" />
                                <Text className="text-xs text-gray-400 mt-1 font-medium">Add Photo</Text>
                            </View>
                        )}
                    </TouchableOpacity>
                </View>

                {/* Full Name */}
                <View className="mb-6">
                    <Text className="text-sm font-bold text-gray-600 mb-2">FULL NAME</Text>
                    <TextInput
                        className={`bg-gray-50 border rounded-xl p-4 text-gray-800 text-base ${isNameTooLong ? 'border-red-500' : 'border-gray-200'
                            }`}
                        placeholder="e.g. John Doe"
                        placeholderTextColor="#9CA3AF"
                        value={fullName}
                        onChangeText={setFullName}
                        maxLength={MAX_NAME_LENGTH + 10}
                    />
                    {isNameTooLong && (
                        <Text className="text-red-500 text-xs mt-1">
                            Name too long ({fullName.length}/{MAX_NAME_LENGTH})
                        </Text>
                    )}
                </View>

                {/* Age and Sex Row */}
                <View className="flex-row mb-6 gap-4">
                    <View className="flex-1">
                        <Text className="text-sm font-bold text-gray-600 mb-2">AGE</Text>
                        <TextInput
                            className={`bg-gray-50 border rounded-xl p-4 text-gray-800 text-base ${isAgeInvalid ? 'border-red-500' : 'border-gray-200'
                                }`}
                            placeholder="Years"
                            placeholderTextColor="#9CA3AF"
                            keyboardType="numeric"
                            maxLength={3}
                            value={age}
                            onChangeText={handleAgeChange}
                        />
                        {isAgeInvalid && (
                            <Text className="text-red-500 text-xs mt-1">Max age is {MAX_AGE}</Text>
                        )}
                    </View>
                    <View className="flex-1">
                        <Text className="text-sm font-bold text-gray-600 mb-2">SEX</Text>
                        <View className="flex-row bg-gray-50 border border-gray-200 rounded-xl overflow-hidden">
                            <TouchableOpacity
                                className={`flex-1 py-4 items-center ${sex === 'Male' ? 'bg-red-600' : 'bg-transparent'}`}
                                onPress={() => setSex('Male')}
                            >
                                <Text className={`font-bold ${sex === 'Male' ? 'text-white' : 'text-gray-500'}`}>Male</Text>
                            </TouchableOpacity>
                            <TouchableOpacity
                                className={`flex-1 py-4 items-center ${sex === 'Female' ? 'bg-red-600' : 'bg-transparent'}`}
                                onPress={() => setSex('Female')}
                            >
                                <Text className={`font-bold ${sex === 'Female' ? 'text-white' : 'text-gray-500'}`}>Female</Text>
                            </TouchableOpacity>
                        </View>
                    </View>
                </View>

                {/* Heart Rate */}
                <View className="mb-6">
                    <Text className="text-sm font-bold text-gray-600 mb-2">HEART RATE (BPM)</Text>
                    <TextInput
                        className="bg-gray-50 border border-gray-200 rounded-xl p-4 text-gray-800 text-base"
                        placeholder="e.g. 72"
                        placeholderTextColor="#9CA3AF"
                        keyboardType="numeric"
                        maxLength={3}
                        value={heartRate}
                        onChangeText={(text) => setHeartRate(text.replace(/[^0-9]/g, ''))}
                    />
                </View>

                {/* Medical History */}
                <View className="mb-6">
                    <Text className="text-sm font-bold text-gray-600 mb-2">MEDICAL HISTORY</Text>
                    <View className="relative">
                        <TextInput
                            className={`bg-gray-50 border rounded-xl p-4 text-gray-800 text-base h-32 ${isHistoryTooLong ? 'border-red-500' : 'border-gray-200'
                                }`}
                            placeholder="Known allergies, previous conditions..."
                            placeholderTextColor="#9CA3AF"
                            multiline
                            textAlignVertical="top"
                            value={history}
                            onChangeText={setHistory}
                        />
                        <Text className={`absolute bottom-2 right-3 text-xs ${isHistoryTooLong ? 'text-red-500' : 'text-gray-400'
                            }`}>
                            {historyCharsLeft} left
                        </Text>
                    </View>
                </View>

            </ScrollView>

            {/* Footer Button */}
            <View className="p-6 bg-white border-t border-gray-100 absolute bottom-0 left-0 right-0">
                <TouchableOpacity
                    className={`py-4 rounded-xl items-center ${(isNameTooLong || isAgeInvalid || isHistoryTooLong)
                            ? 'bg-gray-300'
                            : 'bg-red-600'
                        }`}
                    onPress={handleSave}
                    disabled={isNameTooLong || isAgeInvalid || isHistoryTooLong}
                >
                    <Text className="text-white font-bold text-lg">
                        {editMode ? 'Save Changes' : 'Proceed to Scan'}
                    </Text>
                </TouchableOpacity>
            </View>
        </View>
    );
}
