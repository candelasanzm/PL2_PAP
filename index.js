// Cargar las variables de entorno desde el archivo .env
require('dotenv').config();

// Importar las dependencias necesarias
const express = require('express'); // frmamework para crear el servidor
const path = require('node:path'); // utilidades para trabajar con rutas de archivos
const sql = require('mssql'); // libreria para conectar con SQL Server

const app = express(); // crear la aplicacion Express
const port = process.env.PORT ||3000; //definir el puerto: usa el del entorno si existe, sino usa 3000 por defecto

app.use(express.json()); // middleware permite que express entienda peticiones con cuerpo en formato JSON
app.use(express.static('.')); // middleware sirve archivos estaticos desde la carpeta del proyecto

// Configuracion de la conexion a la base de datos en Azure SQL
const config = {
  user: process.env.DB_USER,
  password: process.env.DB_PASSWORD,
  server: process.env.DB_SERVER,
  database: process.env.DB_NAME,
  options: {
    encrypt: true, // obligatorio en Azure SQL para conexion segura
    trustServerCertificate: false, // forzamos validacion del certificado para mayor seguridad
  }
};

// Endpoint GET sirve el archivo html del visor cuando entras a la URL principal
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'index.html')); // envia el index.html al navegador
});

// Endpoint GET /api/partidas devuelve todas las partidas con sus resultados 
app.get('/api/partidas', async (req, res) => {
  try {
    // Conectar al pool de la base de datos
    let pool = await sql.connect(config);

    // Consultar todas las partidas ordenadas por fecha y hora de la más reciente en adelante
    let result = await pool.request()
      .query('SELECT * FROM partidas ORDER BY fecha_hora DESC');

    // Guardar el array de partidas
    const partidas = result.recordset;

    // Para cada partida buscar sus resultados asociados en partida_resultados
    for (let partida of partidas) {
      let resultados = await pool.request()
        .input('partidaId', sql.Int, partida.id) // pasamos el id de la partida como parametro
        .query('SELECT * FROM partida_resultados WHERE partida_id = @partidaId ORDER BY indice ASC');
      partida.resultados = resultados.recordset; // anidamos los resultados dentro de cada partida
    }

    // Devolver el JSON con todas las partidas y sus resultados
    res.json(partidas);
  } catch (err) {
    // Si hay un error en la consulta, lo mostramos en consola y devolvemos el error 500
    console.error('Error al consultar la base de datos:', err);
    res.status(500).json({
      error: 'Error al consultar la base de datos',
      details: err.message
    });
  }
});

// Endpoint POST /api/partidas recibe una partida desde CUDA y la guarda en la base de datos
app.post('/api/partidas', async (req, res) => {
  try {
    // 1. Leer los datos del cuerpo de la petición (enviado por CUDA en formato JSON)
    const {usuario, fase, opciones_entrada, resultados} = req.body;

    // 2. Validar que los campos obligatorios no sean nulos o vacíos
    if (!usuario || !fase) {
      return res.status(400).json({
        error: 'Los campos "usuario" y "fase" son obligatorios'
      });
    }

    // 3. Conectar al pool de la base de datos
    let pool = await sql.connect(config);

    // 4. Insertar la partida en la tabla principal y obtener el ID generado
    let insertarPartida = await pool.request()
      .input('usuario', sql.NVarChar, usuario.trim()) // limpiamos espacios sobrantes
      .input('fase', sql.NVarChar, fase.trim())
      .input('opciones_entrada', sql.NVarChar, opciones_entrada || '') // si viene vacio guardamos el string vacio
      .query(`
        INSERT INTO partidas (usuario, fase, opciones_entrada)
        OUTPUT INSERTED.id
        VALUES (@usuario, @fase, @opciones_entrada)
      `);
    const partidaId = insertarPartida.recordset[0].id; // ID generado por SQL Server

    // 5. Si vienen resultados en el body, los insertamos uno a uno en partida_resultados
    if (Array.isArray(resultados) && resultados.length > 0) {
      for (let i = 0; i < resultados.length; i++) {
        await pool.request()
          .input('partidaId', sql.Int, partidaId) // ID de la partida creada
          .input('indice', sql.Int, i + 1) // indice del resultado para mantener el orden
          .input('detalle', sql.NVarChar, String(resultados[i])) // contenido del resultado
          .query(`
            INSERT INTO partida_resultados (partida_id, indice, detalle)
            VALUES (@partidaId, @indice, @detalle)
          `);
      }
    }

    // 6. Responder con codigo 201 (creado correctamente)
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

// Arrancar el servidor en el puerto definido y mostrar un mensaje en consola
app.listen(port, () => {
  console.log(`Servidor escuchando en http://localhost:${port}`);
});