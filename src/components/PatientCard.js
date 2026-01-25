import React from 'react';
import { View, Text, Image, StyleSheet, TouchableOpacity } from 'react-native';

export default function PatientCard({ patient, onPress }) {
    const { name, estimated_age, dob, photo_uri, last_scan_result, triage_category } = patient;

    const getBorderColor = () => {
        switch (triage_category) {
            case 'P1': return '#D32F2F'; // Critical
            case 'P2': return '#FBC02D'; // Moderate
            case 'P3': return '#388E3C'; // Stable
            default: return '#ccc';
        }
    };

    const getStatusLabel = () => {
        switch (triage_category) {
            case 'P1': return 'CRITICAL';
            case 'P2': return 'MODERATE';
            case 'P3': return 'STABLE';
            default: return 'UNKNOWN';
        }
    };

    return (
        <TouchableOpacity onPress={onPress}>
            <View style={[styles.card, { borderLeftColor: getBorderColor(), borderLeftWidth: 6 }]}>
                <View style={styles.imageContainer}>
                    {photo_uri ? (
                        <Image source={{ uri: photo_uri }} style={styles.photo} />
                    ) : (
                        <View style={styles.placeholderPhoto}>
                            <Text style={styles.placeholderText}>{name.charAt(0)}</Text>
                        </View>
                    )}
                </View>
                <View style={styles.infoContainer}>
                    <Text style={styles.name}>{name}</Text>
                    <Text style={styles.details}>
                        {estimated_age ? `${estimated_age} yrs` : dob || 'Unknown Age'}
                    </Text>
                    {last_scan_result ? (
                        <Text style={styles.diagnosis} numberOfLines={1}>
                            AI: {last_scan_result}
                        </Text>
                    ) : (
                        <Text style={styles.noScan}>No scan data</Text>
                    )}
                </View>
                <View style={styles.statusContainer}>
                    <Text style={[styles.statusText, { color: getBorderColor() }]}>{getStatusLabel()}</Text>
                </View>
            </View>
        </TouchableOpacity>
    );
}

const styles = StyleSheet.create({
    card: {
        flexDirection: 'row',
        backgroundColor: '#fff',
        marginBottom: 8,
        borderRadius: 8,
        padding: 12,
        elevation: 2,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 1 },
        shadowOpacity: 0.1,
        shadowRadius: 2,
        alignItems: 'center',
    },
    imageContainer: {
        marginRight: 12,
    },
    photo: {
        width: 60,
        height: 60,
        borderRadius: 30,
        backgroundColor: '#eee',
    },
    placeholderPhoto: {
        width: 60,
        height: 60,
        borderRadius: 30,
        backgroundColor: '#e0e0e0',
        justifyContent: 'center',
        alignItems: 'center',
    },
    placeholderText: {
        fontSize: 24,
        fontWeight: 'bold',
        color: '#757575',
    },
    infoContainer: {
        flex: 1,
        justifyContent: 'center',
    },
    name: {
        fontSize: 18,
        fontWeight: 'bold',
        color: '#212121',
        marginBottom: 4,
    },
    details: {
        fontSize: 14,
        color: '#757575',
        marginBottom: 4,
    },
    diagnosis: {
        fontSize: 14,
        color: '#1976D2',
        fontWeight: '500',
    },
    noScan: {
        fontSize: 14,
        color: '#9e9e9e',
        fontStyle: 'italic',
    },
    statusContainer: {
        justifyContent: 'center',
    },
    statusText: {
        fontWeight: 'bold',
        fontSize: 12,
    }
});
