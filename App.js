import * as React from 'react';
import { registerRootComponent } from 'expo';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { StatusBar } from 'expo-status-bar';
import './src/global.css';

import { AppProvider } from './src/context/AppContext';
import DashboardScreen from './src/screens/DashboardScreen';
import PatientIntakeScreen from './src/screens/PatientIntakeScreen';
import PreScanScreen from './src/screens/PreScanScreen';
import ScanResultScreen from './src/screens/ScanResultScreen';
import PatientDetailScreen from './src/screens/PatientDetailScreen';
import HelpScreen from './src/screens/HelpScreen';

const Stack = createNativeStackNavigator();

export default function App() {
  return (
    <AppProvider>
      <NavigationContainer>
        <StatusBar style="dark" />
        <Stack.Navigator
          initialRouteName="Dashboard"
          screenOptions={{
            headerShown: false,
            contentStyle: { backgroundColor: '#F8FAFC' }
          }}
        >
          <Stack.Screen name="Dashboard" component={DashboardScreen} />
          <Stack.Screen name="Intake" component={PatientIntakeScreen} />
          <Stack.Screen name="PreScan" component={PreScanScreen} />
          <Stack.Screen name="Result" component={ScanResultScreen} />
          <Stack.Screen name="Detail" component={PatientDetailScreen} />
          <Stack.Screen name="Help" component={HelpScreen} />
        </Stack.Navigator>
      </NavigationContainer>
    </AppProvider>
  );
}


registerRootComponent(App);
