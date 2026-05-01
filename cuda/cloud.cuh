#ifndef CLOUD_CUH // si no esta definido CLOUD_CUH
#define CLOUD_CUH // lo definimos para evitar inclusiones multiples del mismo archivo

void enviar_al_cloud(const char* fase, const char* opciones, const char* resultadosArray[], int numResultados);
void preguntar_y_enviar(const char* fase, const char* opciones, char resultados[][256], int numResultados);

#endif // fin del bloque ifndef