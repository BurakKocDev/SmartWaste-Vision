allprojects {
    repositories {
        google()
        mavenCentral()
    }
}

val buildDirOverride = providers.environmentVariable("SMARTWASTE_ANDROID_BUILD_DIR").orNull
val newBuildDir: Directory = if (buildDirOverride.isNullOrBlank()) {
    rootProject.layout.buildDirectory.dir("../../build").get()
} else {
    rootProject.layout.projectDirectory.dir(buildDirOverride)
}
rootProject.layout.buildDirectory.value(newBuildDir)

subprojects {
    val newSubprojectBuildDir: Directory = newBuildDir.dir(project.name)
    project.layout.buildDirectory.value(newSubprojectBuildDir)
}
subprojects {
    project.evaluationDependsOn(":app")
}

tasks.register<Delete>("clean") {
    delete(rootProject.layout.buildDirectory)
}
