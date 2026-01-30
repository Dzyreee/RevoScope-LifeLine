import * as SQLite from 'expo-sqlite';

let db;

const getDB = async () => {
  if (!db) {
    db = await SQLite.openDatabaseAsync('revoscope.db');
  }
  return db;
};

export const initDB = async () => {
  const database = await getDB();
  await database.execAsync(`
    PRAGMA foreign_keys = ON;

    CREATE TABLE IF NOT EXISTS patients (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      full_name TEXT NOT NULL,
      age INTEGER,
      sex TEXT,
      history TEXT,
      profile_image TEXT,
      heart_rate INTEGER,
      severity_score INTEGER DEFAULT 0,
      confidence_score INTEGER DEFAULT 0,
      esi_level INTEGER DEFAULT 5,
      triage_advice TEXT,
      created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS scans (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      patient_id INTEGER,
      audio_uri TEXT,
      diagnosis TEXT,
      severity_score INTEGER,
      confidence_score INTEGER,
      esi_level INTEGER,
      status TEXT, -- 'Critical', 'Monitoring', 'Normal'
      heart_rate INTEGER,
      timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
      FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE
    );
  `);
};

export const addPatient = async (patient) => {
  const { full_name, age, sex, history, profile_image, heart_rate, severity_score, confidence_score, esi_level, triage_advice } = patient;
  const database = await getDB();
  const result = await database.runAsync(
    `INSERT INTO patients (full_name, age, sex, history, profile_image, heart_rate, severity_score, confidence_score, esi_level, triage_advice) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);`,
    [full_name, age, sex, history, profile_image, heart_rate || null, severity_score || 0, confidence_score || 0, esi_level || 5, triage_advice || '']
  );
  return result.lastInsertRowId;
};

export const addScan = async (patientId, audioUri, diagnosis, severityScore, confidenceScore, esiLevel, status, triageAdvice = '', heartRate = null) => {
  const database = await getDB();
  // Insert scan
  await database.runAsync(
    `INSERT INTO scans (patient_id, audio_uri, diagnosis, severity_score, confidence_score, esi_level, status, heart_rate) VALUES (?, ?, ?, ?, ?, ?, ?, ?);`,
    [patientId, audioUri, diagnosis, severityScore, confidenceScore, esiLevel, status, heartRate]
  );

  // Update patient's latest scores, triage advice, and heart rate
  await database.runAsync(
    `UPDATE patients SET severity_score = ?, confidence_score = ?, esi_level = ?, triage_advice = ?, heart_rate = ? WHERE id = ?;`,
    [severityScore, confidenceScore, esiLevel, triageAdvice, heartRate, patientId]
  );
};

export const getPatients = async () => {
  const database = await getDB();
  return await database.getAllAsync(`SELECT * FROM patients ORDER BY severity_score DESC, created_at DESC`);
};

export const getPatientHistory = async (patientId) => {
  const database = await getDB();
  const patient = await database.getFirstAsync(`SELECT * FROM patients WHERE id = ?`, [patientId]);
  const scans = await database.getAllAsync(`SELECT * FROM scans WHERE patient_id = ? ORDER BY timestamp DESC`, [patientId]);
  return { patient, scans };
};

export const deletePatient = async (patientId) => {
  const database = await getDB();
  await database.runAsync(`DELETE FROM patients WHERE id = ?`, [patientId]);
  await database.runAsync(`DELETE FROM scans WHERE patient_id = ?`, [patientId]);
};

export const updatePatient = async (patientId, updates) => {
  const { full_name, age, sex, history, profile_image, heart_rate } = updates;
  const database = await getDB();
  await database.runAsync(
    `UPDATE patients SET full_name = ?, age = ?, sex = ?, history = ?, profile_image = ?, heart_rate = ? WHERE id = ?`,
    [full_name, age, sex, history, profile_image, heart_rate || null, patientId]
  );
};


// Summary stats for dashboard
export const getDashboardStats = async () => {
  const database = await getDB();
  // Count based on status in scans? Or based on current severity_score thresholds?
  // Using scans status for now as it's explicit.
  // Actually, dashboard usually counts unique patients in each category.
  // Let's assume patients category is derived from their latest severity.:
  // Critical > 70, Monitoring > 30, Normal <= 30

  const patients = await database.getAllAsync(`SELECT severity_score FROM patients`);

  let critical = 0;
  let monitoring = 0;
  let normal = 0;

  patients.forEach(p => {
    if (p.severity_score >= 70) critical++;
    else if (p.severity_score >= 30) monitoring++;
    else normal++;
  });

  return { total: patients.length, critical, monitoring, normal };
};

export const resetDB = async () => {
  const database = await getDB();
  await database.execAsync(`
    DROP TABLE IF EXISTS scans;
    DROP TABLE IF EXISTS patients;
  `);
  await initDB();
};
