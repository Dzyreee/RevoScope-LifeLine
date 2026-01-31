import React, { useState } from 'react';
import { View, Text, TextInput, TouchableOpacity, ScrollView, Alert, Image, Switch } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import * as ImagePicker from 'expo-image-picker';
import { useApp } from '../context/AppContext';

const MAX_NAME_LENGTH = 100;
const MAX_HISTORY_LENGTH = 1000;
const MAX_AGE = 150;

export default function PatientIntakeScreen({ route, navigation }) {
    const { createPatient, updatePatient, isTestingMode } = useApp();

    const editMode = route.params?.editMode || false;
    const existingData = route.params?.patientData || null;

    const [fullName, setFullName] = useState(existingData?.full_name || '');
    const [age, setAge] = useState(existingData?.age?.toString() || '');
    const [sex, setSex] = useState(existingData?.sex || null);
    const [history, setHistory] = useState(existingData?.history || '');
    const [photo, setPhoto] = useState(existingData?.profile_image || '');
    const [includeHeartRate, setIncludeHeartRate] = useState(false);
    const [forcedResult, setForcedResult] = useState(null); // 'Normal', 'Crackles', 'Wheezing'

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
                include_heart_rate: includeHeartRate // Save preference
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
                navigation.replace('PreScan', { patientId: newId, includeHeartRate, forcedResult });
            }
        } catch (e) {
            console.error(e);
            Alert.alert('Error', 'Could not save patient record.');
        }
    };

    const handleRandomize = () => {
        const randomPatients = [
            { name: 'James Wilson', age: '45', sex: 'Male', history: 'Mild asthma, seasonal allergies. Uses inhaler occasionally.' },
            { name: 'Sarah Connor', age: '32', sex: 'Female', history: 'No clear history. Complains of shortness of breath after exercise.' },
            { name: 'Robert Chen', age: '68', sex: 'Male', history: 'Chronic smoker (20 years), potential COPD signs.' },
            { name: 'Emily Davis', age: '24', sex: 'Female', history: 'Recovering from acute bronchitis. Still has a persistent cough.' },
            { name: 'Michael Jordan', age: '50', sex: 'Male', history: 'High blood pressure, history of pneumonia 5 years ago.' },
            { name: 'Linda Martinez', age: '59', sex: 'Female', history: 'Diabetes Type 2. Reports wheezing at night.' },
            { name: 'David Kim', age: '41', sex: 'Male', history: 'No significant prior history. Recent cold symptoms that worsened.' },
            { name: 'Patricia O\'Neil', age: '75', sex: 'Female', history: 'Congestive heart failure, fluid retention issues.' },
            { name: 'Thomas Anderson', age: '29', sex: 'Male', history: 'Healthy, active. Sudden onset of chest tightness.' },
            { name: 'Jessica Brown', age: '35', sex: 'Female', history: 'Allergic to penicillin. Frequent sinus infections.' }
        ];

        const randomPatient = randomPatients[Math.floor(Math.random() * randomPatients.length)];

        setFullName(randomPatient.name);
        setAge(randomPatient.age);
        setSex(randomPatient.sex);
        setHistory(randomPatient.history);
        setIncludeHeartRate(Math.random() > 0.5);
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
                {/* Heart Rate Scan Toggle */}
                {!editMode && (
                    <View className="mb-6 flex-row items-center justify-between bg-gray-50 p-4 rounded-xl border border-gray-200">
                        <View className="flex-1 mr-4">
                            <Text className="text-sm font-bold text-gray-800 mb-1">Scan Heart Rate?</Text>
                            <Text className="text-xs text-gray-500">
                                Analyze breathing sounds for heart rate estimation (PCG).
                            </Text>
                        </View>
                        <Switch
                            trackColor={{ false: '#D1D5DB', true: '#EF4444' }}
                            thumbColor={'#FFFFFF'}
                            ios_backgroundColor="#D1D5DB"
                            onValueChange={setIncludeHeartRate}
                            value={includeHeartRate}
                        />
                    </View>
                )}

                {/* Randomizer Button (Testing Mode Only) - Moved here for visibility */}
                {isTestingMode && !editMode && (
                    <TouchableOpacity
                        className="mt-2 mb-6 bg-amber-100 p-4 rounded-xl border border-amber-300 flex-row items-center justify-center border-dashed"
                        onPress={handleRandomize}
                    >
                        <Ionicons name="dice-outline" size={24} color="#D97706" />
                        <Text className="text-amber-800 font-bold ml-2">Randomize Patient Data</Text>
                    </TouchableOpacity>
                )}

                {/* Demo Presets (Hidden at bottom) */}
                {!editMode && (
                    <View>
                        {/* Large spacer to hide presets deep at the bottom */}
                        <View style={{ height: 1500 }} />

                        <View className="mt-10 mb-20 border-t border-gray-100 pt-8">
                            <View className="flex-row items-center mb-4 px-2">
                                <Ionicons name="flask-outline" size={20} color="#9CA3AF" />
                                <Text className="text-sm font-bold text-gray-400 ml-2 uppercase tracking-widest">
                                    Demonstration Presets (Internal)
                                </Text>
                            </View>

                            <View className="gap-3">
                                <TouchableOpacity
                                    className={`p-4 rounded-2xl border-2 flex-row items-center justify-between ${forcedResult === 'Normal' ? 'border-green-500 bg-green-50' : 'border-gray-100 bg-gray-50'
                                        }`}
                                    onPress={() => setForcedResult(forcedResult === 'Normal' ? null : 'Normal')}
                                >
                                    <View className="flex-row items-center">
                                        <View className={`h-8 w-8 rounded-full items-center justify-center ${forcedResult === 'Normal' ? 'bg-green-500' : 'bg-gray-200'}`}>
                                            <Ionicons name="checkmark" size={18} color="#fff" />
                                        </View>
                                        <Text className={`ml-3 font-bold ${forcedResult === 'Normal' ? 'text-green-700' : 'text-gray-500'}`}>Result: Normal</Text>
                                    </View>
                                    <Text className="text-[10px] font-bold text-green-600 bg-white px-2 py-1 rounded-full border border-green-100">STABLE</Text>
                                </TouchableOpacity>

                                <TouchableOpacity
                                    className={`p-4 rounded-2xl border-2 flex-row items-center justify-between ${forcedResult === 'Crackles' ? 'border-yellow-500 bg-yellow-50' : 'border-gray-100 bg-gray-50'
                                        }`}
                                    onPress={() => setForcedResult(forcedResult === 'Crackles' ? null : 'Crackles')}
                                >
                                    <View className="flex-row items-center">
                                        <View className={`h-8 w-8 rounded-full items-center justify-center ${forcedResult === 'Crackles' ? 'bg-yellow-500' : 'bg-gray-200'}`}>
                                            <Ionicons name="alert" size={18} color="#fff" />
                                        </View>
                                        <Text className={`ml-3 font-bold ${forcedResult === 'Crackles' ? 'text-yellow-700' : 'text-gray-500'}`}>Result: Crackles</Text>
                                    </View>
                                    <Text className="text-[10px] font-bold text-yellow-600 bg-white px-2 py-1 rounded-full border border-yellow-100">MODERATE</Text>
                                </TouchableOpacity>

                                <TouchableOpacity
                                    className={`p-4 rounded-2xl border-2 flex-row items-center justify-between ${forcedResult === 'Wheezing' ? 'border-red-500 bg-red-50' : 'border-gray-100 bg-gray-50'
                                        }`}
                                    onPress={() => setForcedResult(forcedResult === 'Wheezing' ? null : 'Wheezing')}
                                >
                                    <View className="flex-row items-center">
                                        <View className={`h-8 w-8 rounded-full items-center justify-center ${forcedResult === 'Wheezing' ? 'bg-red-500' : 'bg-gray-200'}`}>
                                            <Ionicons name="warning" size={18} color="#fff" />
                                        </View>
                                        <Text className={`ml-3 font-bold ${forcedResult === 'Wheezing' ? 'text-red-700' : 'text-gray-500'}`}>Result: Wheezing</Text>
                                    </View>
                                    <Text className="text-[10px] font-bold text-red-600 bg-white px-2 py-1 rounded-full border border-red-100">CRITICAL</Text>
                                </TouchableOpacity>
                            </View>
                            <Text className="text-gray-400 text-[10px] text-center mt-4 italic font-medium">
                                Selecting a preset will bypass real AI analysis for demonstration purposes.
                            </Text>
                        </View>
                    </View>
                )}



            </ScrollView>

            {/* Footer Button */}
            <View className="p-6 bg-white border-t border-gray-100 absolute bottom-0 left-0 right-0">
                <TouchableOpacity
                    style={styles.shadow}
                    className={`py-4 rounded-xl items-center flex-row justify-center ${(isNameTooLong || isAgeInvalid || isHistoryTooLong)
                        ? 'bg-gray-300'
                        : 'bg-red-600'
                        }`}
                    onPress={handleSave}
                    disabled={isNameTooLong || isAgeInvalid || isHistoryTooLong}
                >
                    <Text className="text-white font-bold text-lg mr-2">
                        {editMode ? 'Save Changes' : 'Proceed to Scan'}
                    </Text>
                    {!editMode && <Ionicons name="arrow-forward" size={20} color="#fff" />}
                </TouchableOpacity>
            </View>
        </View>
    );
}

const styles = {
    shadow: {
        shadowColor: '#EF4444',
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.3,
        shadowRadius: 8,
        elevation: 6,
    }
};
