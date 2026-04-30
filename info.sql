CREATE TABLE partidas (
    id INT IDENTITY(1,1) PRIMARY KEY,
    usuario NVARCHAR(100) NOT NULL,
    fecha_hora DATETIME NOT NULL DEFAULT GETDATE(),
    fase NVARCHAR(100) NOT NULL,
    opciones_entrada NVARCHAR(MAX),
    resultado NVARCHAR(MAX)
);