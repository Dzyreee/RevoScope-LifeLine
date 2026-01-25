import * as SQLite from 'expo-sqlite';

let db;

const getDB = async () => {
  if (!db) {
    db = await SQLite.openDatabaseAsync('soundscope.db');
  }
  return db;
};

export const initDB = async () => {
  const database = await getDB();
  await database.execAsync(`
    CREATE TABLE IF NOT EXISTS patients (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      name TEXT NOT NULL,
      photo_uri TEXT,
      dob TEXT,
      estimated_age INTEGER,
      chief_complaint TEXT,
      triage_category TEXT DEFAULT 'P3',
      last_scan_result TEXT,
      created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
  `);
};

export const addPatient = async (patient) => {
  const { name, photo_uri, dob, estimated_age, chief_complaint, triage_category, last_scan_result } = patient;
  const database = await getDB();
  const result = await database.runAsync(
    `INSERT INTO patients (name, photo_uri, dob, estimated_age, chief_complaint, triage_category, last_scan_result) VALUES (?, ?, ?, ?, ?, ?, ?);`,
    [name, photo_uri, dob, estimated_age, chief_complaint, triage_category || 'P3', last_scan_result || '']
  );
  return result;
};

export const getPatients = async () => {
  const database = await getDB();
  const allRows = await database.getAllAsync(
    `SELECT * FROM patients ORDER BY created_at DESC;`
  );
  return allRows;
};

export const deletePatient = async (id) => {
  const database = await getDB();
  const result = await database.runAsync(
    `DELETE FROM patients WHERE id = ?;`,
    [id]
  );
  return result;
};



export const updatePatientCategory = async (id, category) => {
  const database = await getDB();
  const result = await database.runAsync(
    `UPDATE patients SET triage_category = ? WHERE id = ?;`,
    [category, id]
  );
  return result;
};
