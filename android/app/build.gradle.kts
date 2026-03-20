plugins {
    id("com.android.application")
    id("kotlin-android")
    id("dev.flutter.flutter-gradle-plugin")
}

android {
    namespace  = "com.example.adas_499"
    compileSdk = 36
    ndkVersion = "27.0.12077973"

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = JavaVersion.VERSION_17.toString()
        // Enable optimisations for the Kotlin compiler
        freeCompilerArgs += listOf("-opt-in=kotlin.RequiresOptIn")
    }

    defaultConfig {
        applicationId = "com.example.adas_499"
        minSdk    = 26   // GPU delegate requires API 26+; NNAPI requires API 27+
        targetSdk = 35
        versionCode = flutter.versionCode
        versionName = flutter.versionName

        // Build only for 64-bit ARM (arm64-v8a) and legacy 32-bit (armeabi-v7a).
        // Dropping x86/x86_64 cuts APK size and avoids slow emulator stubs for
        // the TFLite GPU/NNAPI delegates.
        ndk {
            abiFilters += listOf("arm64-v8a", "armeabi-v7a")
        }
    }

    // ── Prevent model files from being compressed ─────────────────────────────
    // TFLite needs to mmap() these directly; compression breaks mmap().
    androidResources {
        noCompress += listOf("tflite", "lite", "bin")
    }

    buildTypes {
        release {
            // Enable R8 / ProGuard for smaller, faster release builds.
            isMinifyEnabled   = true
            isShrinkResources = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
            signingConfig = signingConfigs.getByName("debug")
        }
        debug {
            // Keep debug builds fast — no obfuscation.
            isMinifyEnabled = false
        }
    }

    packaging {
        jniLibs {
            // Retain native .so files required by TFLite GPU / NNAPI delegates.
            keepDebugSymbols += listOf("**/libtensorflowlite_gpu_jni.so",
                                       "**/libtensorflowlite_nnapi_jni.so")
        }
        resources {
            // Strip duplicate licence files to keep APK clean.
            excludes += listOf("META-INF/LICENSE*", "META-INF/NOTICE*")
        }
    }
}

flutter {
    source = "../.."
}

dependencies {
    // ── CameraX ───────────────────────────────────────────────────────────────
    // Use a BOM so all CameraX artefacts are version-locked together.
    val cameraxVersion = "1.4.2"
    implementation("androidx.camera:camera-core:$cameraxVersion")
    implementation("androidx.camera:camera-camera2:$cameraxVersion")
    implementation("androidx.camera:camera-lifecycle:$cameraxVersion")
    // camera-view is NOT needed — we render preview via Flutter Texture.

    // ── TFLite runtime (native Android SDK — NOT the tflite_flutter plugin) ───
    // Using the official Google SDK gives us direct access to GPU & NNAPI
    // delegates without any Dart overhead in the hot path.
    implementation("org.tensorflow:tensorflow-lite:2.17.0")
    implementation("org.tensorflow:tensorflow-lite-gpu:2.17.0")
    // Remove the problematic delegate plugin and use the standard GPU delegate
    // implementation("org.tensorflow:tensorflow-lite-gpu-delegate-plugin:0.4.4")
    // NNAPI delegate (ships inside tensorflow-lite but explicit dep ensures it)
    implementation("org.tensorflow:tensorflow-lite-api:2.17.0")

    // ── AndroidX ──────────────────────────────────────────────────────────────
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.7")
}
