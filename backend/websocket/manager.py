"""
websocket/manager.py
====================
Gestor de conexiones WebSocket por sala de reunión.
"""

import json
import asyncio
from typing import Dict, List, Optional
from fastapi import WebSocket
from datetime import datetime


class ConexionCliente:
    def __init__(self, websocket: WebSocket, usuario_id: int, nombre: str, sala_id: str):
        self.ws           = websocket
        self.usuario_id   = usuario_id
        self.nombre       = nombre
        self.sala_id      = sala_id
        self.conectado_en = datetime.now()
        self.activo       = True


class GestorWebSocket:
    def __init__(self):
        self.salas: Dict[str, List[ConexionCliente]] = {}
        self._lock = asyncio.Lock()

    async def conectar(self, websocket: WebSocket, sala_id: str,
                       usuario_id: int, nombre: str) -> ConexionCliente:
        await websocket.accept()
        cliente = ConexionCliente(websocket, usuario_id, nombre, sala_id)

        async with self._lock:
            if sala_id not in self.salas:
                self.salas[sala_id] = []
            self.salas[sala_id].append(cliente)

        await self.broadcast_sistema(sala_id, f"{nombre} se unió a la sala")

        participantes = self.obtener_participantes(sala_id)
        await self._enviar(cliente, {
            "tipo": "sala_info",
            "participantes": participantes,
            "sala_id": sala_id,
        })

        print(f"[WS] {nombre} conectado a sala '{sala_id}'")
        return cliente

    async def desconectar(self, cliente: ConexionCliente) -> None:
        cliente.activo = False
        sala_id = cliente.sala_id

        async with self._lock:
            if sala_id in self.salas:
                self.salas[sala_id] = [
                    c for c in self.salas[sala_id] if c.usuario_id != cliente.usuario_id
                ]
                if not self.salas[sala_id]:
                    del self.salas[sala_id]

        await self.broadcast_sistema(sala_id, f"{cliente.nombre} salió de la sala")
        print(f"[WS] {cliente.nombre} desconectado de sala '{sala_id}'")

    async def _enviar(self, cliente: ConexionCliente, datos: dict) -> bool:
        if not cliente.activo:
            return False
        try:
            await cliente.ws.send_json(datos)
            return True
        except Exception as e:
            cliente.activo = False
            print(f"[WS] Error enviando a {cliente.nombre}: {e}")
            return False

    async def broadcast(self, sala_id: str, datos: dict,
                        excluir_id: Optional[int] = None) -> int:
        if sala_id not in self.salas:
            return 0
        enviados = 0
        for cliente in list(self.salas[sala_id]):
            if excluir_id and cliente.usuario_id == excluir_id:
                continue
            if await self._enviar(cliente, datos):
                enviados += 1
        return enviados

    async def broadcast_sistema(self, sala_id: str, mensaje: str,
                                 excluir_id: Optional[int] = None) -> None:
        await self.broadcast(sala_id, {
            "tipo":      "sistema",
            "mensaje":   mensaje,
            "timestamp": datetime.now().isoformat(),
        }, excluir_id=excluir_id)

    async def enviar_traduccion(self, sala_id: str, usuario_id: int,
                                 nombre_usuario: str, letra_actual: str,
                                 palabra_actual: str, frase_completa: str,
                                 confianza: float) -> None:
        await self.broadcast(sala_id, {
            "tipo":           "traduccion_live",
            "usuario_id":     usuario_id,
            "nombre":         nombre_usuario,
            "letra_actual":   letra_actual,
            "palabra_actual": palabra_actual,
            "frase_completa": frase_completa,
            "confianza":      round(confianza, 3),
            "timestamp":      datetime.now().isoformat(),
        }, excluir_id=usuario_id)

    async def enviar_mensaje_chat(self, sala_id: str, usuario_id: int,
                                   nombre_usuario: str, texto: str,
                                   texto_corregido: str, mensaje_id: int) -> None:
        await self.broadcast(sala_id, {
            "tipo":            "mensaje_chat",
            "mensaje_id":      mensaje_id,
            "usuario_id":      usuario_id,
            "nombre":          nombre_usuario,
            "texto":           texto,
            "texto_corregido": texto_corregido,
            "timestamp":       datetime.now().isoformat(),
        })

    async def enviar_señal_webrtc(self, sala_id: str, de_usuario_id: int,
                                   para_usuario_id: int, señal: dict) -> None:
        if sala_id not in self.salas:
            return
        for cliente in self.salas[sala_id]:
            if cliente.usuario_id == para_usuario_id:
                await self._enviar(cliente, {
                    "tipo":  "webrtc_señal",
                    "de":    de_usuario_id,
                    "señal": señal,
                })
                return

    def obtener_participantes(self, sala_id: str) -> List[dict]:
        if sala_id not in self.salas:
            return []
        return [
            {
                "usuario_id":   c.usuario_id,
                "nombre":       c.nombre,
                "conectado_en": c.conectado_en.isoformat(),
            }
            for c in self.salas[sala_id] if c.activo
        ]

    def sala_existe(self, sala_id: str) -> bool:
        return sala_id in self.salas and len(self.salas[sala_id]) > 0

    def total_conexiones(self) -> int:
        return sum(len(c) for c in self.salas.values())

    def stats(self) -> dict:
        return {
            "salas_activas":    len(self.salas),
            "total_conexiones": self.total_conexiones(),
            "salas": {s: len(c) for s, c in self.salas.items()}
        }


# Instancia global
ws_manager = GestorWebSocket()
