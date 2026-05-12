import React, { useState } from 'react';
import { StyleSheet, Text, View, Image, TouchableOpacity, ActivityIndicator, Alert } from 'react-native';
import * as ImagePicker from 'expo-image-picker';

const BACKEND_URL = process.env.EXPO_PUBLIC_BACKEND_URL;

export default function App() {
  const[imageUri, setImageUri] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const pickImage = async () => {
    let permissionResult = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permissionResult.granted) {
      Alert.alert("Brak uprawnień", "Wymagany dostęp do galerii.");
      return;
    }

    let pickerResult = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect:[1, 1],
      quality: 0.8,
    });

    if (!pickerResult.canceled) {
      setImageUri(pickerResult.assets[0].uri);
      setResult(null); // Resetuj poprzedni wynik
    }
  };

  const takePhoto = async () => {
    let permissionResult = await ImagePicker.requestCameraPermissionsAsync();
    if (!permissionResult.granted) {
      Alert.alert("Brak uprawnień", "Wymagany dostęp do aparatu.");
      return;
    }

    let cameraResult = await ImagePicker.launchCameraAsync({
      allowsEditing: true,
      aspect: [1, 1],
      quality: 0.8,
    });

    if (!cameraResult.canceled) {
      setImageUri(cameraResult.assets[0].uri);
      setResult(null);
    }
  };

  const analyzeImage = async () => {
    if (!imageUri) {
      Alert.alert("Błąd", "Najpierw wybierz lub zrób zdjęcie!");
      return;
    }

    setLoading(true);

    try {
      let localUri = imageUri;
      let filename = localUri.split('/').pop();
      let match = /\.(\w+)$/.exec(filename);
      let type = match ? `image/${match[1]}` : `image`;

      let formData = new FormData();
      formData.append('file', { uri: localUri, name: filename, type });

      const response = await fetch(BACKEND_URL, {
        method: 'POST',
        body: formData,
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      const responseJson = await response.json();
      setResult(responseJson);

    } catch (error) {
      console.error(error);
      Alert.alert("Błąd serwera", "Nie udało się połączyć z backendem.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>AI Skin Scanner</Text>

      {/* Podgląd zdjęcia */}
      <View style={styles.imageContainer}>
        {imageUri ? (
          <Image source={{ uri: imageUri }} style={styles.image} />
        ) : (
          <Text style={styles.placeholderText}>Brak zdjęcia</Text>
        )}
      </View>

      {/* Przyciski wyboru zdjęcia */}
      <View style={styles.buttonRow}>
        <TouchableOpacity style={styles.buttonSecondary} onPress={takePhoto}>
          <Text style={styles.buttonText}>Aparat</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.buttonSecondary} onPress={pickImage}>
          <Text style={styles.buttonText}>Galeria</Text>
        </TouchableOpacity>
      </View>

      {/* Przycisk analizy */}
      <TouchableOpacity style={styles.buttonPrimary} onPress={analyzeImage} disabled={loading}>
        {loading ? <ActivityIndicator color="#fff" /> : <Text style={styles.buttonText}>Analizuj Zmianę</Text>}
      </TouchableOpacity>

      {/* Wyświetlanie wyniku */}
      {result && (
        <View style={styles.resultContainer}>
          <Text style={styles.resultLabel}>Wynik Diagnozy:</Text>
          <Text style={[styles.predictionText, result.prediction === "malignant" ? styles.danger : styles.safe]}>
            {result.prediction === "malignant" ? "POTENCJALNY CZERNIAK" : "ZMIANA ŁAGODNA"}
          </Text>
          <Text style={styles.confidenceText}>
            Pewność modelu: {(result.confidence * 100).toFixed(1)}%
          </Text>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F5F7FA',
    alignItems: 'center',
    paddingTop: 80,
    paddingHorizontal: 20,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#2C3E50',
    marginBottom: 30,
  },
  imageContainer: {
    width: 250,
    height: 250,
    backgroundColor: '#E1E8ED',
    borderRadius: 20,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 20,
    overflow: 'hidden',
  },
  image: {
    width: '100%',
    height: '100%',
  },
  placeholderText: {
    color: '#7F8C8D',
    fontSize: 16,
  },
  buttonRow: {
    flexDirection: 'row',
    gap: 15,
    marginBottom: 20,
  },
  buttonSecondary: {
    backgroundColor: '#34495E',
    paddingVertical: 12,
    paddingHorizontal: 25,
    borderRadius: 10,
  },
  buttonPrimary: {
    backgroundColor: '#3498DB',
    paddingVertical: 15,
    paddingHorizontal: 40,
    borderRadius: 10,
    width: '100%',
    alignItems: 'center',
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
  resultContainer: {
    marginTop: 40,
    padding: 20,
    backgroundColor: 'white',
    borderRadius: 15,
    width: '100%',
    alignItems: 'center',
    shadowColor: '#000',
    shadowOpacity: 0.1,
    shadowRadius: 10,
    elevation: 5,
  },
  resultLabel: {
    fontSize: 14,
    color: '#7F8C8D',
    marginBottom: 5,
  },
  predictionText: {
    fontSize: 22,
    fontWeight: '900',
    marginBottom: 5,
  },
  danger: {
    color: '#E74C3C', 
  },
  safe: {
    color: '#2ECC71', 
  },
  confidenceText: {
    fontSize: 16,
    color: '#34495E',
  }
});