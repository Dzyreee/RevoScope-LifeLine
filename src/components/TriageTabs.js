import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';

export default function TriageTabs({ selectedTab, onTabSelect, counts }) {
    const tabs = [
        { key: 'P1', label: 'CRITICAL', color: '#D32F2F', count: counts?.P1 || 0 },
        { key: 'P2', label: 'MODERATE', color: '#FBC02D', count: counts?.P2 || 0 },
        { key: 'P3', label: 'STABLE', color: '#388E3C', count: counts?.P3 || 0 },
    ];

    return (
        <View style={styles.container}>
            {tabs.map((tab) => {
                const isActive = selectedTab === tab.key;
                return (
                    <TouchableOpacity
                        key={tab.key}
                        style={[
                            styles.tab,
                            isActive && { backgroundColor: tab.color + '20', borderColor: tab.color }, // 20 opacity hex
                        ]}
                        onPress={() => onTabSelect(tab.key)}
                    >
                        <Text style={[styles.label, isActive && { color: tab.color }]}>
                            {tab.label}
                        </Text>
                        <View style={[styles.badge, isActive ? { backgroundColor: tab.color } : { backgroundColor: '#e0e0e0' }]}>
                            <Text style={[styles.count, isActive ? { color: '#fff' } : { color: '#757575' }]}>
                                {tab.count}
                            </Text>
                        </View>
                    </TouchableOpacity>
                );
            })}
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        flexDirection: 'row',
        padding: 8,
        backgroundColor: '#fff',
        borderBottomWidth: 1,
        borderBottomColor: '#e0e0e0',
    },
    tab: {
        flex: 1,
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        paddingVertical: 12,
        borderBottomWidth: 3,
        borderBottomColor: 'transparent',
        borderRadius: 4,
    },
    label: {
        fontSize: 14,
        fontWeight: 'bold',
        color: '#757575',
        marginRight: 6,
    },
    badge: {
        paddingHorizontal: 6,
        paddingVertical: 2,
        borderRadius: 12,
    },
    count: {
        fontSize: 12,
        fontWeight: 'bold',
    },
});
