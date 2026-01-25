import * as React from 'react';
import { registerRootComponent } from 'expo';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import DashboardScreen from './src/screens/DashboardScreen';
import ImagingScreen from './src/screens/ImagingScreen';
import PatientIntakeScreen from './src/screens/PatientIntakeScreen';
import { StatusBar } from 'expo-status-bar';
import { AppProvider } from './src/context/AppContext';

const Stack = createNativeStackNavigator();

export default function App() {
  return (
    <AppProvider>
      <NavigationContainer>
        <StatusBar style="dark" />
        <Stack.Navigator
          initialRouteName="Dashboard"
          screenOptions={{
            headerStyle: { backgroundColor: '#f4f4f5' },
            headerTintColor: '#09090b',
            headerTitleStyle: { fontWeight: 'bold' },
          }}
        >
          <Stack.Screen
            name="Dashboard"
            component={DashboardScreen}
            options={{ title: 'SoundScope Triage' }}
          />
          <Stack.Screen
            name="Intake"
            component={PatientIntakeScreen}
            options={{ title: 'New Patient Entry' }}
          />
          <Stack.Screen
            name="Imaging"
            component={ImagingScreen}
            options={{ title: 'Live Diagnostic View' }}
          />
        </Stack.Navigator>
      </NavigationContainer>
    </AppProvider>
  );
}

registerRootComponent(App);
