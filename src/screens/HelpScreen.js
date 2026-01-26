import React from 'react';
import { View, Text, ScrollView, TouchableOpacity, Image } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { StatusBar } from 'expo-status-bar';

const logoImage = require('../assets/logo.png');

const steps = [
    {
        number: 1,
        title: 'Add Patient',
        description: 'Start by adding a new patient with their basic information, or select an existing patient from the dashboard.',
        icon: 'person-add',
        color: '#3B82F6'
    },
    {
        number: 2,
        title: 'Position Stethoscope',
        description: 'Place your digital stethoscope firmly on the patient\'s chest. Ensure good contact for clear audio capture.',
        icon: 'medical',
        color: '#10B981'
    },
    {
        number: 3,
        title: 'Record Breath Sounds',
        description: 'Tap "New Scan" to begin a 15-second recording. Keep the stethoscope still and ask the patient to breathe normally.',
        icon: 'mic',
        color: '#F59E0B'
    },
    {
        number: 4,
        title: 'AI Analysis',
        description: 'Our AI model analyzes the audio to detect crackles, wheezes, and other abnormal respiratory sounds.',
        icon: 'analytics',
        color: '#8B5CF6'
    },
    {
        number: 5,
        title: 'Review Results',
        description: 'View the diagnosis, severity score, and ESI triage level. Follow the clinical recommendations provided.',
        icon: 'clipboard',
        color: '#DC2626'
    }
];

const tips = [
    { icon: 'volume-mute', text: 'Ensure a quiet environment for accurate recordings' },
    { icon: 'time', text: 'Record for the full 15 seconds for best results' },
    { icon: 'fitness', text: 'Ask patient to breathe deeply and slowly' },
    { icon: 'refresh', text: 'Multiple scans help track changes over time' }
];

export default function HelpScreen({ navigation }) {
    return (
        <View className="flex-1 bg-white">
            <StatusBar style="dark" />

            {/* Header */}
            <View className="pt-16 px-6 pb-4 bg-white border-b border-gray-100 flex-row justify-between items-center">
                <TouchableOpacity
                    className="h-11 w-11 bg-gray-100 rounded-full items-center justify-center"
                    onPress={() => navigation.goBack()}
                >
                    <Ionicons name="arrow-back" size={24} color="#374151" />
                </TouchableOpacity>
                <Text className="text-xl font-bold text-gray-800">How to Use</Text>
                <View className="w-11" />
            </View>

            <ScrollView className="flex-1" contentContainerStyle={{ paddingBottom: 40 }}>
                {/* Hero Section */}
                <View className="items-center py-8 px-6">
                    <Image source={logoImage} style={{ width: 200, height: 60 }} resizeMode="contain" />
                    <Text className="text-gray-500 text-center mt-4 text-base">
                        AI-powered respiratory sound analysis for rapid triage decisions
                    </Text>
                </View>

                {/* Steps */}
                <View className="px-6">
                    <Text className="text-lg font-bold text-gray-800 mb-4">Getting Started</Text>

                    {steps.map((step, index) => (
                        <View key={step.number} className="mb-4">
                            <View className="bg-white rounded-2xl border border-gray-100 p-4 shadow-sm">
                                <View className="flex-row items-start">
                                    <View
                                        className="h-12 w-12 rounded-xl items-center justify-center mr-4"
                                        style={{ backgroundColor: `${step.color}15` }}
                                    >
                                        <Ionicons name={step.icon} size={24} color={step.color} />
                                    </View>
                                    <View className="flex-1">
                                        <View className="flex-row items-center mb-1">
                                            <View className="h-6 w-6 bg-red-600 rounded-full items-center justify-center mr-2">
                                                <Text className="text-white font-bold text-sm">{step.number}</Text>
                                            </View>
                                            <Text className="text-gray-800 font-bold text-base">{step.title}</Text>
                                        </View>
                                        <Text className="text-gray-500 text-sm leading-5">{step.description}</Text>
                                    </View>
                                </View>
                            </View>
                            {index < steps.length - 1 && (
                                <View className="items-center py-1">
                                    <Ionicons name="chevron-down" size={20} color="#D1D5DB" />
                                </View>
                            )}
                        </View>
                    ))}
                </View>

                {/* Tips Section */}
                <View className="px-6 mt-4">
                    <Text className="text-lg font-bold text-gray-800 mb-4">Tips for Best Results</Text>

                    <View className="bg-amber-50 rounded-2xl p-4 border border-amber-200">
                        {tips.map((tip, index) => (
                            <View key={index} className={`flex-row items-center ${index < tips.length - 1 ? 'mb-3' : ''}`}>
                                <View className="h-8 w-8 bg-amber-100 rounded-full items-center justify-center mr-3">
                                    <Ionicons name={tip.icon} size={16} color="#F59E0B" />
                                </View>
                                <Text className="text-amber-800 flex-1 text-sm">{tip.text}</Text>
                            </View>
                        ))}
                    </View>
                </View>

                {/* Understanding Results */}
                <View className="px-6 mt-6">
                    <Text className="text-lg font-bold text-gray-800 mb-4">Understanding Results</Text>

                    <View className="bg-gray-50 rounded-2xl p-4">
                        <View className="flex-row items-center mb-3">
                            <View className="h-4 w-4 rounded-full bg-green-500 mr-3" />
                            <Text className="text-gray-600 flex-1">
                                <Text className="font-bold">Normal</Text> - Clear breath sounds, no intervention needed
                            </Text>
                        </View>
                        <View className="flex-row items-center mb-3">
                            <View className="h-4 w-4 rounded-full bg-yellow-500 mr-3" />
                            <Text className="text-gray-600 flex-1">
                                <Text className="font-bold">Monitoring</Text> - Mild abnormalities detected
                            </Text>
                        </View>
                        <View className="flex-row items-center">
                            <View className="h-4 w-4 rounded-full bg-red-500 mr-3" />
                            <Text className="text-gray-600 flex-1">
                                <Text className="font-bold">Critical</Text> - Urgent evaluation required
                            </Text>
                        </View>
                    </View>
                </View>

                {/* Got It Button */}
                <View className="px-6 mt-8">
                    <TouchableOpacity
                        className="bg-red-600 py-4 rounded-xl items-center"
                        onPress={() => navigation.goBack()}
                    >
                        <Text className="text-white font-bold text-lg">Got It!</Text>
                    </TouchableOpacity>
                </View>
            </ScrollView>
        </View>
    );
}
