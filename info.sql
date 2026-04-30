CREATE TABLE partidas (
    id INT IDENTITY(1,1) PRIMARY KEY,
    usuario NVARCHAR(100) NOT NULL,
    fecha_hora DATETIME NOT NULL DEFAULT GETDATE(),
    fase NVARCHAR(100) NOT NULL,
    opciones_entrada NVARCHAR(MAX),
    resultado NVARCHAR(MAX)
);

CREATE TABLE partida_resultados (
    id INT IDENTITY(1,1) PRIMARY KEY,
    partida_id INT NOT NULL,
    indice INT NOT NULL,
    detalle NVARCHAR(MAX) NOT NULL,
    CONSTRAINT fk_partida_resultados FOREIGN KEY (partida_id) REFERENCES partidas(id) ON DELETE CASCADE
);

-- Creamos un índice para acelerar las consultas por partida
CREATE INDEX index_partida_resultados_partida_id ON partida_resultados(partida_id);