plugins {
    id("com.android.application")
    id("kotlin-android")
    id("dev.flutter.flutter-gradle-plugin")
}

android {
    namespace = "com.example.adas_499"
    compileSdk = 36
    ndkVersion = "27.0.12077973"

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = JavaVersion.VERSION_17.toString()
    }

    defaultConfig {
        applicationId = "com.example.adas_499"
        minSdk = 26          // NNAPI requires API 27; GPU delegate requires 26+
        targetSdk = 35
        versionCode = flutter.versionCode
        versionName = flutter.versionName

        // Prevent TFLite model files from being compressed in the APK
        ndk {
            abiFilters += listOf("arm64-v8a", "armeabi-v7a")
        }
    }

    aaptOptions {
        noCompress += listOf("tflite", "lite")
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            signingConfig = signingConfigs.getByName("debug")
        }
    }

    packaging {
        jniLibs {
            // Keep native .so files for TFLite delegates
            keepDebugSymbols += "**/*.so"
        }
    }
}

flutter {
    source = "../.."
}

dependencies {
    // ── CameraX ───────────────────────────────────────────────────────────────
    val cameraxVersion = "1.4.2"
    implementation("androidx.camera:camera-core:$cameraxVersion")
    implementation("androidx.camera:camera-camera2:$cameraxVersion")
    implementation("androidx.camera:camera-lifecycle:$cameraxVersion")

    // ── TFLite (native Android SDK — NOT tflite_flutter) ─────────────────────
    implementation("org.tensorflow:tensorflow-lite:2.17.0")
    implementation("org.tensorflow:tensorflow-lite-gpu:2.17.0")
    implementation("org.tensorflow:tensorflow-lite-gpu-delegate-plugin:0.4.4")

    // ── AndroidX core ─────────────────────────────────────────────────────────
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.7")
}
