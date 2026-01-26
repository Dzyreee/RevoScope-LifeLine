/** @type {import('tailwindcss').Config} */
module.exports = {
    content: ["./App.{js,jsx,ts,tsx}", "./src/**/*.{js,jsx,ts,tsx}"],
    presets: [require("nativewind/preset")],
    theme: {
        extend: {
            colors: {
                primary: "#DC2626",        // Medical Red
                background: "#FFFFFF",
                surface: "#FEF2F2",         // Light red tint
                success: "#10B981",
                warning: "#F59E0B",
                error: "#DC2626",
                critical: "#DC2626",        // Red
                monitoring: "#F59E0B",      // Amber
                normal: "#10B981",          // Green
                onPrimary: "#FFFFFF",
                text: "#1F2937",
                textSecondary: "#6B7280",
                border: "#E5E7EB",
            },
            fontSize: {
                'title': ['28px', '34px'],
                'heading': ['24px', '30px'],
                'body': ['16px', '24px'],
            }
        },
    },
    plugins: [],
}
