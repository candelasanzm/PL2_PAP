require('dotenv').config();

const express = require('express');
const path = require('node:path');
const sql = require('mssql');

const app = express();
const port = process.env.PORT ||3000;

app.use(express.json());
app.use(express.static('.'));

const config = {
  user: process.env.DB_USER,
  password: process.env.DB_PASSWORD,
  server: process.env.DB_SERVER,
  database: process.env.DB_NAME,
  options: {
    encrypt: true,
    trustServerCertificate: false,
  }
};

app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'index.html'));
});

app.get('/api/partidas', async (req, res) => {
  try {
    let pool = await sql.connect(config);
    let result = await pool.request()
      .query('SELECT * FROM partidas ORDER BY fecha_hora DESC');;
    res.json(result.recordset);
  } catch (err) {
    console.error('Error al consultar la base de datos:', err);
    res.status(500).json({
      error: 'Error al consultar la base de datos',
      details: err.message
    });
  }
});

app.post('/api/partidas', async (req, res) => {
  try {
    // 1. Leer datos del body
    const {usuario, fase, opciones_entrada, resultado} = req.body;

    // 2. Validar que los campos obligatorios no sean nulos
    if (!usuario || !fase) {
      return res.status(400).json({
        error: 'Los campos "usuario" y "fase" son obligatorios'
      });
    }

    // 3. Conectar a la BD e insertar
    let pool = await sql.connect(config);
    await pool.request()
      .input('usuario', sql.NVarChar, usuario.trim())
      .input('fase', sql.NVarChar, fase.trim())
      .input('opciones_entrada', sql.NVarChar, opciones_entrada || '')
      .input('resultado', sql.NVarChar, resultado || '')
      .query(`
        INSERT INTO partidas (usuario, fase, opciones_entrada, resultado)
        VALUES (@usuario, @fase, @opciones_entrada, @resultado)
      `);

    // 4. Responder con éxito
    res.status(201).json({
      message: 'Partida registrada exitosamente'
    });
  } catch (err) {
    console.error('Error al insertar en la base de datos:', err);
    res.status(500).json({
      error: 'Error al guardar la partida',
      details: err.message
    });
  }
});

app.listen(port, () => {
  console.log(`Servidor escuchando en http://localhost:${port}`);
});