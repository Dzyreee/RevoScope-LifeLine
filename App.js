import * as React from 'react';
import { registerRootComponent } from 'expo';
import { View, Text, StyleSheet } from 'react-native';

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error("ErrorBoundary caught an error", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <View style={styles.errorContainer}>
          <Text style={styles.errorTitle}>Something went wrong</Text>
          <Text style={styles.errorText}>{this.state.error && this.state.error.toString()}</Text>
        </View>
      );
    }

    return this.props.children;
  }
}

const styles = StyleSheet.create({
  errorContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
    backgroundColor: '#fff',
  },
  errorTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 10,
    color: 'red',
  },
  errorText: {
    fontSize: 16,
    color: '#333',
    textAlign: 'center',
  },
});

import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { StatusBar } from 'expo-status-bar';
import './src/global.css';
import { useState, useEffect } from 'react';
import { ActivityIndicator } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';

import { AppProvider } from './src/context/AppContext';
import DashboardScreen from './src/screens/DashboardScreen';
import PatientIntakeScreen from './src/screens/PatientIntakeScreen';
import PreScanScreen from './src/screens/PreScanScreen';
import ScanResultScreen from './src/screens/ScanResultScreen';
import PatientDetailScreen from './src/screens/PatientDetailScreen';
import HelpScreen from './src/screens/HelpScreen';
import WelcomeScreen from './src/screens/WelcomeScreen';
import LoginScreen from './src/screens/LoginScreen';

const Stack = createNativeStackNavigator();

export default function App() {
  const [initialRoute, setInitialRoute] = useState(null);

  useEffect(() => {
    (async () => {
      try {
        const token = await AsyncStorage.getItem('userToken');
        setInitialRoute(token ? 'Dashboard' : 'Welcome');
      } catch (e) {
        setInitialRoute('Welcome');
      }
    })();
  }, []);

  if (!initialRoute) {
    return (
      <View style={{ flex: 1, alignItems: 'center', justifyContent: 'center' }}>
        <ActivityIndicator size="large" />
      </View>
    );
  }

  return (
    <ErrorBoundary>
      <AppProvider>
        <NavigationContainer>
          <StatusBar style="dark" />
          <Stack.Navigator
            initialRouteName={initialRoute}
            screenOptions={{
              headerShown: false,
              contentStyle: { backgroundColor: '#FFF8F8' }
            }}
          >
            <Stack.Screen name="Welcome" component={WelcomeScreen} />
            <Stack.Screen name="Login" component={LoginScreen} />
            <Stack.Screen name="Dashboard" component={DashboardScreen} />
            <Stack.Screen name="Intake" component={PatientIntakeScreen} />
            <Stack.Screen name="PreScan" component={PreScanScreen} />
            <Stack.Screen name="Result" component={ScanResultScreen} />
            <Stack.Screen name="Detail" component={PatientDetailScreen} />
            <Stack.Screen name="Help" component={HelpScreen} />
          </Stack.Navigator>
        </NavigationContainer>
      </AppProvider>
    </ErrorBoundary>
  );
}


registerRootComponent(App);
