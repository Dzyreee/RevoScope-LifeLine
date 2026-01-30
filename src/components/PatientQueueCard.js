import React from 'react';
import { View, Text, Image, TouchableOpacity } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { ESI_CONFIG } from '../constants/Config';

export default function PatientQueueCard({ patient, onPress }) {
    const { full_name, profile_image, severity_score, confidence_score, heart_rate, esi_level } = patient;

    // Severity color logic
    let severityColor = "#10B981"; // Green
    let severityBg = "bg-emerald-50";
    if (severity_score >= 70) {
        severityColor = "#DC2626";
        severityBg = "bg-red-50";
    } else if (severity_score >= 30) {
        severityColor = "#F59E0B";
        severityBg = "bg-amber-50";
    }

    return (
        <TouchableOpacity
            onPress={onPress}
            className="flex-row items-center p-4 mb-3 bg-white rounded-2xl border border-gray-100 shadow-sm"
        >
            {/* Profile Image */}
            <View className="h-14 w-14 rounded-full bg-gray-100 overflow-hidden items-center justify-center mr-4">
                {profile_image ? (
                    <Image source={{ uri: profile_image }} className="h-full w-full" />
                ) : (
                    <Ionicons name="person" size={24} color="#9CA3AF" />
                )}
            </View>

            {/* Info */}
            <View className="flex-1">
                <Text className="text-lg font-bold text-gray-800">{full_name}</Text>
                <View className="flex-row items-center mt-1 gap-2">
                    {esi_level && (
                        <View className="px-2 py-0.5 rounded-full" style={{ backgroundColor: ESI_CONFIG[esi_level]?.color || '#9CA3AF' }}>
                            <Text className="text-[10px] font-black text-white">ESI {esi_level}</Text>
                        </View>
                    )}
                    <View className="bg-gray-100 px-2 py-0.5 rounded-full">
                        <Text className="text-xs font-medium text-gray-500">Conf: {confidence_score}%</Text>
                    </View>
                    {heart_rate && (
                        <View className="bg-blue-100 px-2 py-0.5 rounded-full flex-row items-center gap-1">
                            <Ionicons name="heart" size={10} color="#3B82F6" />
                            <Text className="text-xs font-medium text-blue-600">{heart_rate} BPM</Text>
                        </View>
                    )}
                </View>
            </View>

            {/* Severity Score */}
            <View className={`items-center px-4 py-2 rounded-xl ${severityBg}`}>
                <Text className="text-xs font-bold text-gray-500 mb-0.5">SEVERITY</Text>
                <Text className="text-2xl font-bold" style={{ color: severityColor }}>
                    {severity_score}%
                </Text>
            </View>
        </TouchableOpacity>
    );
}
