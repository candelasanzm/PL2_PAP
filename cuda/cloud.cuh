#ifndef CLOUD_CUH
#define CLOUD_CUH

void enviar_al_cloud(const char* fase, const char* opciones, const char* resultadosArray[], int numResultados);
void preguntar_y_enviar(const char* fase, const char* opciones, char resultados[][256], int numResultados);

#endif