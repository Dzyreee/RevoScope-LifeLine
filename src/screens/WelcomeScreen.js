import React, { useEffect, useRef } from 'react';
import {
    View,
    Text,
    TouchableOpacity,
    StyleSheet,
    Image,
    Animated,
    Dimensions,
    StatusBar,
} from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';

const { width, height } = Dimensions.get('window');

export default function WelcomeScreen({ navigation }) {
    const fadeAnim = useRef(new Animated.Value(0)).current;
    const slideAnim = useRef(new Animated.Value(50)).current;
    const buttonFadeAnim = useRef(new Animated.Value(0)).current;
    const pulseAnim = useRef(new Animated.Value(1)).current;

    useEffect(() => {
        // Entrance animations
        Animated.sequence([
            Animated.parallel([
                Animated.timing(fadeAnim, {
                    toValue: 1,
                    duration: 800,
                    useNativeDriver: true,
                }),
                Animated.timing(slideAnim, {
                    toValue: 0,
                    duration: 800,
                    useNativeDriver: true,
                }),
            ]),
            Animated.timing(buttonFadeAnim, {
                toValue: 1,
                duration: 500,
                useNativeDriver: true,
            }),
        ]).start();

        // Subtle pulse animation for logo
        Animated.loop(
            Animated.sequence([
                Animated.timing(pulseAnim, {
                    toValue: 1.02,
                    duration: 2000,
                    useNativeDriver: true,
                }),
                Animated.timing(pulseAnim, {
                    toValue: 1,
                    duration: 2000,
                    useNativeDriver: true,
                }),
            ])
        ).start();
    }, []);

    const handleConnectOnline = () => {
        navigation.navigate('Login');
    };

    const handleConnectOffline = async () => {
        try {
            await AsyncStorage.setItem('userToken', JSON.stringify({ guest: true, offline: true }));
            navigation.reset({ index: 0, routes: [{ name: 'Dashboard' }] });
        } catch (e) {
            console.error('Failed to save offline mode');
        }
    };

    return (
        <View style={styles.container}>
            <StatusBar barStyle="dark-content" backgroundColor="#FFF5F5" />

            {/* Background decoration */}
            <View style={styles.backgroundDecoration}>
                <View style={[styles.circle, styles.circle1]} />
                <View style={[styles.circle, styles.circle2]} />
                <View style={[styles.circle, styles.circle3]} />
            </View>

            {/* Logo and Title Section */}
            <Animated.View
                style={[
                    styles.logoContainer,
                    {
                        opacity: fadeAnim,
                        transform: [
                            { translateY: slideAnim },
                            { scale: pulseAnim },
                        ],
                    },
                ]}
            >
                <View style={[styles.logoWrapper, { paddingHorizontal: 32, paddingVertical: 24 }]}>
                    <Text style={styles.textLogo}>
                        Revo<Text style={{ color: '#E85656' }}>Scope</Text>
                    </Text>
                </View>
                <Text style={styles.tagline}>Advanced Medical Imaging Analysis</Text>
            </Animated.View>

            {/* Buttons Section */}
            <Animated.View
                style={[
                    styles.buttonContainer,
                    { opacity: buttonFadeAnim },
                ]}
            >
                {/* Connect Online Button */}
                <TouchableOpacity
                    style={styles.primaryButton}
                    onPress={handleConnectOnline}
                    activeOpacity={0.85}
                >
                    <LinearGradient
                        colors={['#E85656', '#D44848']}
                        start={{ x: 0, y: 0 }}
                        end={{ x: 1, y: 1 }}
                        style={styles.gradientButton}
                    >
                        <Ionicons name="cloud-outline" size={24} color="#fff" style={styles.buttonIcon} />
                        <View style={styles.buttonTextContainer}>
                            <Text style={styles.primaryButtonText}>Connect Online</Text>
                            <Text style={styles.buttonSubtext}>Sync & backup your data</Text>
                        </View>
                        <Ionicons name="chevron-forward" size={20} color="#fff" />
                    </LinearGradient>
                </TouchableOpacity>

                {/* Connect Offline Button */}
                <TouchableOpacity
                    style={styles.secondaryButton}
                    onPress={handleConnectOffline}
                    activeOpacity={0.85}
                >
                    <Ionicons name="phone-portrait-outline" size={24} color="#E85656" style={styles.buttonIcon} />
                    <View style={styles.buttonTextContainer}>
                        <Text style={styles.secondaryButtonText}>Connect Offline</Text>
                        <Text style={styles.buttonSubtextSecondary}>Use without an account</Text>
                    </View>
                    <Ionicons name="chevron-forward" size={20} color="#E85656" />
                </TouchableOpacity>
            </Animated.View>

            {/* Footer */}
            <Animated.View style={[styles.footer, { opacity: buttonFadeAnim }]}>
                <Text style={styles.footerText}>
                    By continuing, you agree to our Terms of Service
                </Text>
            </Animated.View>
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#FFF8F8',
        alignItems: 'center',
        justifyContent: 'center',
        paddingHorizontal: 24,
    },
    backgroundDecoration: {
        position: 'absolute',
        width: '100%',
        height: '100%',
    },
    circle: {
        position: 'absolute',
        borderRadius: 999,
        backgroundColor: 'rgba(232, 86, 86, 0.05)',
    },
    circle1: {
        width: 300,
        height: 300,
        top: -100,
        right: -100,
    },
    circle2: {
        width: 200,
        height: 200,
        bottom: 100,
        left: -80,
    },
    circle3: {
        width: 150,
        height: 150,
        top: '40%',
        right: -50,
    },
    logoContainer: {
        alignItems: 'center',
        marginBottom: 60,
    },
    logoWrapper: {
        backgroundColor: '#fff',
        borderRadius: 24,
        padding: 20,
        shadowColor: '#E85656',
        shadowOffset: { width: 0, height: 8 },
        shadowOpacity: 0.15,
        shadowRadius: 24,
        elevation: 8,
        marginBottom: 20,
    },
    textLogo: {
        fontSize: 42,
        fontWeight: '800',
        color: '#1F2937',
        letterSpacing: -1,
    },
    tagline: {
        fontSize: 16,
        color: '#6B7280',
        fontWeight: '500',
        letterSpacing: 0.5,
    },
    buttonContainer: {
        width: '100%',
        gap: 16,
    },
    primaryButton: {
        borderRadius: 16,
        shadowColor: '#E85656',
        shadowOffset: { width: 0, height: 8 },
        shadowOpacity: 0.35,
        shadowRadius: 16,
        elevation: 8,
    },
    gradientButton: {
        flexDirection: 'row',
        alignItems: 'center',
        paddingVertical: 20,
        paddingHorizontal: 24,
        borderRadius: 16,
    },
    secondaryButton: {
        flexDirection: 'row',
        alignItems: 'center',
        paddingVertical: 20,
        paddingHorizontal: 24,
        backgroundColor: '#fff',
        borderRadius: 16,
        borderWidth: 2,
        borderColor: '#E85656',
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.08,
        shadowRadius: 12,
        elevation: 4,
    },
    buttonIcon: {
        marginRight: 16,
    },
    buttonTextContainer: {
        flex: 1,
    },
    primaryButtonText: {
        fontSize: 18,
        fontWeight: '700',
        color: '#fff',
        marginBottom: 2,
    },
    secondaryButtonText: {
        fontSize: 18,
        fontWeight: '700',
        color: '#E85656',
        marginBottom: 2,
    },
    buttonSubtext: {
        fontSize: 13,
        color: 'rgba(255,255,255,0.85)',
    },
    buttonSubtextSecondary: {
        fontSize: 13,
        color: '#9CA3AF',
    },
    footer: {
        position: 'absolute',
        bottom: 40,
        alignItems: 'center',
    },
    footerText: {
        fontSize: 12,
        color: '#9CA3AF',
        textAlign: 'center',
    },
});
