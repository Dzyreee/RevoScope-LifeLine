import React from 'react';
import { View, Text, ScrollView, TouchableOpacity, Image, Dimensions } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { StatusBar } from 'expo-status-bar';
import { LinearGradient } from 'expo-linear-gradient';

const logoImage = require('../assets/logo.png');

const steps = [
    {
        number: 1,
        title: 'Add Patient Profile',
        description: 'Start by creating a new patient record or selecting one from the dashboard.',
        icon: 'person-add-outline',
        color: ['#3B82F6', '#2563EB'], // Blue
        iconColor: '#2563EB'
    },
    {
        number: 2,
        title: 'Position Stethoscope',
        description: 'Place the digital stethoscope firmly on the patient\'s upper back (left or right side). Ensure skin contact.',
        icon: 'medkit-outline',
        color: ['#10B981', '#059669'], // Emerald
        iconColor: '#059669'
    },
    {
        number: 3,
        title: 'Record & Breathe',
        description: 'Tap "New Scan". Instruct patient to breathe deeply for 15 seconds. Follow the breathing guide.',
        icon: 'mic-outline',
        color: ['#F59E0B', '#D97706'], // Amber
        iconColor: '#D97706'
    },
    {
        number: 4,
        title: 'AI Diagnostics',
        description: 'Our advanced Neural Network analyzes the audio for crackles, wheezes, and abnormalities.',
        icon: 'pulse-outline',
        color: ['#8B5CF6', '#7C3AED'], // Violet
        iconColor: '#7C3AED'
    },
    {
        number: 5,
        title: 'Clinical Results',
        description: 'Review the ESI Triage Level, severity score, and automated clinical recommendations.',
        icon: 'document-text-outline',
        color: ['#EF4444', '#DC2626'], // Red
        iconColor: '#DC2626'
    }
];

const bestPractices = [
    { icon: 'volume-mute-outline', title: 'Quiet Room', text: 'Minimize background noise' },
    { icon: 'shirt-outline', title: 'Skin Contact', text: 'Avoid recording through clothes' },
    { icon: 'fitness-outline', title: 'Deep Breaths', text: 'Steady, deep breathing' },
];

export default function HelpScreen({ navigation }) {
    return (
        <View className="flex-1 bg-white">
            <StatusBar style="dark" />

            {/* Header */}
            <View className="pt-16 px-6 pb-4 bg-white border-b border-gray-100 flex-row justify-between items-center z-10">
                <TouchableOpacity
                    className="h-10 w-10 bg-gray-50 rounded-full items-center justify-center border border-gray-100 shadow-sm"
                    onPress={() => navigation.goBack()}
                >
                    <Ionicons name="close" size={24} color="#374151" />
                </TouchableOpacity>
                <Text className="text-lg font-bold text-gray-800">User Guide</Text>
                <View className="w-10" />
            </View>

            <ScrollView className="flex-1" contentContainerStyle={{ paddingBottom: 100 }} showsVerticalScrollIndicator={false}>
                {/* Hero Section */}
                <View className="items-center py-8 px-6">
                    <Image source={logoImage} style={{ width: 400, height: 150, marginBottom: 12 }} resizeMode="contain" />
                    <Text className="text-gray-500 text-center text-sm px-8">
                        Master the RevoScope workflow in 5 simple steps.
                    </Text>
                </View>

                {/* Steps Timeline */}
                <View className="px-6">
                    {steps.map((step, index) => (
                        <View key={step.number} className="flex-row">
                            {/* Timeline Line */}
                            <View className="items-center mr-4">
                                <View className={`h-8 w-8 rounded-full items-center justify-center border-2 border-white shadow-sm z-10`} style={{ backgroundColor: step.iconColor }}>
                                    <Text className="text-white font-bold text-xs">{step.number}</Text>
                                </View>
                                {index < steps.length - 1 && (
                                    <View className="w-[2px] flex-1 bg-gray-100 my-1" />
                                )}
                            </View>

                            {/* Card content */}
                            <View className="flex-1 pb-8">
                                <LinearGradient
                                    colors={['#FFFFFF', '#F9FAFB']}
                                    className="p-5 rounded-2xl border border-gray-100 shadow-sm"
                                >
                                    <View className="flex-row items-center mb-3">
                                        <View className="p-2 rounded-lg bg-gray-50 mr-3">
                                            <Ionicons name={step.icon} size={22} color={step.iconColor} />
                                        </View>
                                        <Text className="text-base font-bold text-gray-800 flex-1">
                                            {step.title}
                                        </Text>
                                    </View>
                                    <Text className="text-gray-500 text-sm leading-6">
                                        {step.description}
                                    </Text>
                                </LinearGradient>
                            </View>
                        </View>
                    ))}
                </View>

                {/* Best Practices Grid */}
                <View className="px-6 mt-4">
                    <Text className="text-base font-bold text-gray-800 mb-4 px-2">Pro Tips for Accuracy</Text>
                    <View className="flex-row flex-wrap gap-3">
                        {bestPractices.map((tip, index) => (
                            <View key={index} className="flex-1 min-w-[30%] bg-blue-50/50 p-3 rounded-xl border border-blue-100 items-center">
                                <Ionicons name={tip.icon} size={20} color="#3B82F6" style={{ marginBottom: 6 }} />
                                <Text className="text-xs font-bold text-blue-900 text-center mb-1">{tip.title}</Text>
                                <Text className="text-[10px] text-blue-700 text-center leading-3">{tip.text}</Text>
                            </View>
                        ))}
                    </View>
                </View>

                {/* Legend */}
                <View className="px-6 mt-8">
                    <View className="bg-gray-900 rounded-3xl p-6 overflow-hidden relative">
                        {/* Decorative blobs */}
                        <View className="absolute top-0 right-0 w-32 h-32 bg-gray-800 rounded-full -mr-16 -mt-16 opacity-50" />
                        <View className="absolute bottom-0 left-0 w-24 h-24 bg-gray-800 rounded-full -ml-12 -mb-12 opacity-50" />

                        <Text className="text-white font-bold text-lg mb-4 z-10">Understanding Triage Results</Text>

                        <View className="flex-row items-center mb-3 z-10">
                            <View className="h-3 w-3 rounded-full bg-green-400 mr-3 shadow-sm shadow-green-400" />
                            <Text className="text-gray-300 text-sm flex-1">
                                <Text className="text-white font-bold">Normal (ESI 4-5)</Text> • Clear/Safe
                            </Text>
                        </View>
                        <View className="flex-row items-center mb-3 z-10">
                            <View className="h-3 w-3 rounded-full bg-yellow-400 mr-3 shadow-sm shadow-yellow-400" />
                            <Text className="text-gray-300 text-sm flex-1">
                                <Text className="text-white font-bold">Monitoring (ESI 3)</Text> • Follow-up
                            </Text>
                        </View>
                        <View className="flex-row items-center z-10">
                            <View className="h-3 w-3 rounded-full bg-red-500 mr-3 shadow-sm shadow-red-500" />
                            <Text className="text-gray-300 text-sm flex-1">
                                <Text className="text-white font-bold">Critical (ESI 1-2)</Text> • Immediate Action
                            </Text>
                        </View>
                    </View>
                </View>

                {/* CTA */}
                <View className="px-6 mt-10">
                    <TouchableOpacity
                        className="bg-red-600 py-4 rounded-xl items-center shadow-lg shadow-red-200"
                        onPress={() => navigation.goBack()}
                    >
                        <Text className="text-white font-bold text-lg">Return to Dashboard</Text>
                    </TouchableOpacity>
                </View>
            </ScrollView>
        </View>
    );
}
