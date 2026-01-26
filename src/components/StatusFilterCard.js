import React from 'react';
import { View, Text, TouchableOpacity } from 'react-native';

export default function StatusFilterCard({ type, count, active, onPress }) {
    // Shorten labels for display
    const displayLabels = {
        Critical: 'CRITICAL',
        Monitoring: 'MONITOR',
        Normal: 'NORMAL'
    };

    // Color config per type
    const colorConfig = {
        Critical: {
            border: 'border-red-500',
            bg: 'bg-red-500',
            text: 'text-red-600',
            activeText: 'text-white'
        },
        Monitoring: {
            border: 'border-amber-500',
            bg: 'bg-amber-500',
            text: 'text-amber-600',
            activeText: 'text-white'
        },
        Normal: {
            border: 'border-emerald-500',
            bg: 'bg-emerald-500',
            text: 'text-emerald-600',
            activeText: 'text-white'
        }
    };

    const colors = colorConfig[type] || colorConfig.Normal;
    const label = displayLabels[type] || type.toUpperCase();

    return (
        <TouchableOpacity
            onPress={onPress}
            className={`flex-1 p-4 mx-1.5 rounded-2xl border-2 ${active
                    ? `${colors.bg} border-transparent`
                    : `bg-white ${colors.border}`
                }`}
        >
            <Text
                className={`text-xs font-bold tracking-wider ${active ? colors.activeText : colors.text
                    }`}
                numberOfLines={1}
            >
                {label}
            </Text>
            <Text className={`text-4xl font-bold mt-1 ${active ? colors.activeText : 'text-gray-800'
                }`}>
                {count}
            </Text>
        </TouchableOpacity>
    );
}
