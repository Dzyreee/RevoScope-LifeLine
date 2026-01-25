import React, { useState } from 'react';
import { View, StyleSheet, FlatList, TouchableOpacity, Text } from 'react-native';
import { useApp } from '../context/AppContext';
import PatientCard from '../components/PatientCard';
import TriageTabs from '../components/TriageTabs';
import DeviceSetupModal from '../components/DeviceSetupModal';
import { Ionicons } from '@expo/vector-icons'; // Ensure @expo/vector-icons works or use standard icons

export default function DashboardScreen({ navigation }) {
    const { patients, isLoading } = useApp();
    const [selectedTab, setSelectedTab] = useState('P3'); // Default to Stable/All or one of them? Prompt says "Three distinct high-contrast sections". Tabs work well.
    const [isSetupVisible, setIsSetupVisible] = useState(false);

    // Derive counts
    const counts = {
        P1: patients.filter(p => p.triage_category === 'P1').length,
        P2: patients.filter(p => p.triage_category === 'P2').length,
        P3: patients.filter(p => p.triage_category === 'P3').length, // Stable is P3
    };

    const filteredPatients = patients.filter(p => p.triage_category === selectedTab);

    if (isLoading) {
        return (
            <View style={styles.loading}>
                <Text>Loading Triage Data...</Text>
            </View>
        );
    }

    return (
        <View style={styles.container}>
            <TriageTabs
                selectedTab={selectedTab}
                onTabSelect={setSelectedTab}
                counts={counts}
            />

            <FlatList
                data={filteredPatients}
                keyExtractor={item => item.id.toString()}
                renderItem={({ item }) => (
                    <PatientCard
                        patient={item}
                        onPress={() => navigation.navigate('Imaging', { patientId: item.id })}
                    />
                )}
                contentContainerStyle={styles.listContent}
                ListEmptyComponent={
                    <View style={styles.emptyState}>
                        <Text style={styles.emptyText}>No patients in {selectedTab}</Text>
                    </View>
                }
            />

            <TouchableOpacity
                style={styles.fab}
                onPress={() => navigation.navigate('Intake')}
            >
                <Text style={styles.fabText}>+</Text>
            </TouchableOpacity>

            {/* Settings / Device Button in Header usually, but adding a float or corner button here for demo */}
            <TouchableOpacity
                style={styles.setupButton}
                onPress={() => setIsSetupVisible(true)}
            >
                <Text style={styles.setupButtonText}>⚙️ Audio Input</Text>
            </TouchableOpacity>

            <DeviceSetupModal
                visible={isSetupVisible}
                onClose={() => setIsSetupVisible(false)}
            />
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#f5f5f5',
    },
    loading: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
    },
    listContent: {
        padding: 16,
        paddingBottom: 80,
    },
    emptyState: {
        padding: 24,
        alignItems: 'center',
    },
    emptyText: {
        color: '#757575',
        fontSize: 16,
    },
    fab: {
        position: 'absolute',
        bottom: 24,
        right: 24,
        width: 56,
        height: 56,
        borderRadius: 28,
        backgroundColor: '#1976D2',
        justifyContent: 'center',
        alignItems: 'center',
        elevation: 4,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.25,
        shadowRadius: 3.84,
    },
    fabText: {
        color: '#fff',
        fontSize: 32,
        marginTop: -4,
    },
    setupButton: {
        position: 'absolute',
        bottom: 24,
        left: 24,
        backgroundColor: '#fff',
        paddingHorizontal: 16,
        paddingVertical: 10,
        borderRadius: 24,
        elevation: 3,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 1 },
        shadowOpacity: 0.2,
        shadowRadius: 1.41,
    },
    setupButtonText: {
        fontWeight: '600',
        color: '#333',
    }

});
