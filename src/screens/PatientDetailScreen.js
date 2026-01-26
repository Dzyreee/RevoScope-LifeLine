import React, { useEffect, useState } from 'react';
import { View, Text, TouchableOpacity, ScrollView, Image, Alert, Modal } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useApp } from '../context/AppContext';

export default function PatientDetailScreen({ route, navigation }) {
    const { patientId } = route.params;
    const { getHistory, deletePatient, refreshDashboard } = useApp();
    const [history, setHistory] = useState(null);
    const [showPhotoModal, setShowPhotoModal] = useState(false);

    useEffect(() => {
        loadData();
    }, []);

    const loadData = async () => {
        const data = await getHistory(patientId);
        setHistory(data);
    };

    const handleEdit = () => {
        navigation.navigate('Intake', {
            editMode: true,
            patientData: history.patient
        });
    };

    const handleDelete = () => {
        Alert.alert(
            "Delete Patient",
            "Are you sure? This action cannot be undone.",
            [
                { text: "Cancel", style: "cancel" },
                {
                    text: "Delete",
                    style: "destructive",
                    onPress: async () => {
                        await deletePatient(patientId);
                        await refreshDashboard();
                        navigation.navigate('Dashboard');
                    }
                }
            ]
        );
    };

    const handleNewScan = () => {
        navigation.navigate('PreScan', { patientId });
    };

    if (!history) return <View className="flex-1 bg-white" />;

    const { patient, scans } = history;
    const { full_name, age, sex, history: medicalHistory, profile_image, severity_score, confidence_score } = patient;

    // Severity Color Logic
    let severityColor = "#10B981";
    let severityBg = "bg-emerald-50";
    let severityBorder = "border-emerald-200";
    if (severity_score >= 70) {
        severityColor = "#DC2626";
        severityBg = "bg-red-50";
        severityBorder = "border-red-200";
    } else if (severity_score >= 30) {
        severityColor = "#F59E0B";
        severityBg = "bg-amber-50";
        severityBorder = "border-amber-200";
    }

    // Get potential condition from last scan
    const lastScan = scans.length > 0 ? scans[0] : null;
    const getPotentialCondition = (diagnosis) => {
        if (!diagnosis) return null;
        const diag = diagnosis.toLowerCase();
        if (diag.includes('absent') || diag.includes('abnormal')) {
            return { condition: 'Possible Pneumothorax (Collapsed Lung)', color: '#DC2626', bg: 'bg-red-50' };
        } else if (diag.includes('crackle') && diag.includes('wheeze')) {
            return { condition: 'Possible Bronchopneumonia', color: '#EA580C', bg: 'bg-orange-50' };
        } else if (diag.includes('crackle')) {
            return { condition: 'Possible Pneumonia or Pulmonary Edema', color: '#F59E0B', bg: 'bg-amber-50' };
        } else if (diag.includes('wheeze')) {
            return { condition: 'Possible Asthma or COPD Exacerbation', color: '#F59E0B', bg: 'bg-amber-50' };
        }
        return null;
    };
    const potentialCondition = lastScan ? getPotentialCondition(lastScan.diagnosis) : null;

    return (
        <View className="flex-1 bg-gray-50">
            {/* Header */}
            <View className="pt-16 pb-4 px-6 bg-white border-b border-gray-100 flex-row justify-between items-center">
                <TouchableOpacity onPress={() => navigation.goBack()} className="p-1">
                    <Ionicons name="arrow-back" size={24} color="#374151" />
                </TouchableOpacity>
                <Text className="text-xl font-bold text-gray-800">Patient Profile</Text>
                <View className="flex-row items-center gap-4">
                    <TouchableOpacity onPress={handleEdit}>
                        <Ionicons name="create-outline" size={24} color="#374151" />
                    </TouchableOpacity>
                    <TouchableOpacity onPress={handleDelete}>
                        <Ionicons name="trash-outline" size={24} color="#DC2626" />
                    </TouchableOpacity>
                </View>
            </View>

            <ScrollView className="flex-1" contentContainerStyle={{ paddingBottom: 100 }}>
                {/* Profile Card */}
                <View className="mx-4 mt-4 p-6 bg-white rounded-2xl border border-gray-100 items-center">
                    <TouchableOpacity
                        onPress={() => profile_image && setShowPhotoModal(true)}
                        activeOpacity={profile_image ? 0.7 : 1}
                    >
                        <View className="h-24 w-24 rounded-full bg-gray-100 mb-4 overflow-hidden border-4 border-gray-50 items-center justify-center">
                            {profile_image ? (
                                <Image source={{ uri: profile_image }} className="h-full w-full" />
                            ) : (
                                <Ionicons name="person" size={48} color="#9CA3AF" />
                            )}
                        </View>
                    </TouchableOpacity>
                    {profile_image && (
                        <Text className="text-xs text-gray-400 mb-2">Tap photo to enlarge</Text>
                    )}
                    <Text className="text-2xl font-bold text-gray-800">{full_name}</Text>
                    <Text className="text-gray-500 font-medium text-base mt-1">
                        {age} Years • {sex || 'Unknown'}
                    </Text>
                </View>

                {/* Scores Row */}
                <View className="flex-row mx-4 mt-4 gap-4">
                    <View className={`flex-1 p-4 rounded-xl border ${severityBorder} ${severityBg} items-center`}>
                        <Text className="text-xs font-bold uppercase tracking-wider mb-1 text-gray-500">Severity</Text>
                        <Text className="text-3xl font-bold" style={{ color: severityColor }}>{severity_score}%</Text>
                    </View>
                    <View className="flex-1 p-4 rounded-xl bg-white border border-gray-100 items-center">
                        <Text className="text-xs font-bold uppercase tracking-wider mb-1 text-gray-400">Confidence</Text>
                        <Text className="text-3xl font-bold text-gray-700">{confidence_score}%</Text>
                    </View>
                    {heart_rate && (
                        <View className="flex-1 p-4 rounded-xl bg-blue-50 border border-blue-200 items-center">
                            <Text className="text-xs font-bold uppercase tracking-wider mb-1 text-blue-500">Heart Rate</Text>
                            <Text className="text-3xl font-bold text-blue-600">{heart_rate}</Text>
                            <Text className="text-xs text-blue-500">BPM</Text>
                        </View>
                    )}

                {/* New Scan Button */}
                <View className="mx-4 mt-4">
                    <TouchableOpacity
                        className="bg-red-600 py-4 rounded-xl items-center flex-row justify-center"
                        onPress={handleNewScan}
                    >
                        <Ionicons name="mic" size={22} color="#fff" />
                        <Text className="text-white font-bold text-lg ml-2">New Scan</Text>
                    </TouchableOpacity>
                </View>

                {/* Medical History */}
                <View className="mx-4 mt-6">
                    <Text className="text-sm font-bold text-gray-500 uppercase tracking-wider mb-2">Medical History</Text>
                    <View className="bg-white p-4 rounded-xl border border-gray-100">
                        <Text className="text-gray-600 leading-6 text-base">{medicalHistory || "None recorded."}</Text>
                    </View>
                </View>

                {/* Potential Condition */}
                {potentialCondition && (
                    <View className="mx-4 mt-6">
                        <Text className="text-sm font-bold text-gray-500 uppercase tracking-wider mb-2">Potential Condition</Text>
                        <View className={`p-4 rounded-xl border ${potentialCondition.bg}`} style={{ borderColor: potentialCondition.color + '40' }}>
                            <View className="flex-row items-center">
                                <View className="h-10 w-10 rounded-full items-center justify-center mr-3" style={{ backgroundColor: potentialCondition.color + '20' }}>
                                    <Ionicons name="alert-circle" size={22} color={potentialCondition.color} />
                                </View>
                                <Text className="text-base font-bold flex-1" style={{ color: potentialCondition.color }}>
                                    {potentialCondition.condition}
                                </Text>
                            </View>
                        </View>
                    </View>
                )}

                {/* Triage Advice */}
                <View className="mx-4 mt-6">
                    <Text className="text-sm font-bold text-gray-500 uppercase tracking-wider mb-2">Triage Recommendation</Text>
                    <View className="bg-red-600 p-5 rounded-xl">
                        <View className="flex-row items-center mb-2">
                            <Ionicons name="medkit" size={18} color="#fff" />
                            <Text className="text-white font-bold ml-2">Action Plan</Text>
                        </View>
                        <Text className="text-red-100 leading-6 text-base">
                            {patient.triage_advice || "Complete a scan to generate recommendations."}
                        </Text>
                    </View>
                </View>

                {/* Scan History */}
                <View className="mx-4 mt-6">
                    <Text className="text-sm font-bold text-gray-500 uppercase tracking-wider mb-3">Previous Scans</Text>
                    {scans.length === 0 ? (
                        <View className="bg-white p-6 rounded-xl border border-gray-100 items-center">
                            <Ionicons name="document-outline" size={32} color="#9CA3AF" />
                            <Text className="text-gray-400 mt-2">No previous scans</Text>
                        </View>
                    ) : (
                        scans.map(scan => (
                            <View key={scan.id} className="bg-white p-4 mb-3 rounded-xl border border-gray-100 flex-row justify-between items-center">
                                <View>
                                    <Text className="font-bold text-gray-700 text-base">{scan.diagnosis}</Text>
                                    <Text className="text-xs text-gray-400 mt-1">{new Date(scan.timestamp).toLocaleString()}</Text>
                                </View>
                                <Text className="font-bold text-gray-500 text-lg">{scan.severity_score}%</Text>
                            </View>
                        ))
                    )}
                </View>
            </ScrollView>

            {/* Photo Modal */}
            <Modal visible={showPhotoModal} transparent animationType="fade">
                <View className="flex-1 bg-black/90 justify-center items-center">
                    <TouchableOpacity
                        className="absolute top-16 right-6 z-10 h-12 w-12 bg-white/20 rounded-full items-center justify-center"
                        onPress={() => setShowPhotoModal(false)}
                    >
                        <Ionicons name="close" size={28} color="#fff" />
                    </TouchableOpacity>
                    {profile_image && (
                        <Image
                            source={{ uri: profile_image }}
                            style={{ width: '90%', height: '70%' }}
                            resizeMode="contain"
                        />
                    )}
                    <Text className="text-white mt-4 text-lg font-medium">{full_name}</Text>
                </View>
            </Modal>
        </View>
    );
}
