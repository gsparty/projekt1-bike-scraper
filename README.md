# Projekt 1 Python

## Übersicht


| Eigenes Projekt |
| Datenherkunft HTML (via Web Scraping von Tutti.ch) | [URL](https://www.tutti.ch/de/q/motorraeder/Ak8CrbW90b3JjeWNsZXOUwMDAwA?sorting=newest&page=2) |
| ML-Algorithmus Random Forest Classifier |
| Repo [URL](https://github.com/gsparty/projekt1-bike-scraper) |

## Dokumentation

### Data Scraping

•  Zweck: Das Scraping-Modul ruft Motorradanzeigen von Tutti.ch ab, extrahiert relevante Parameter wie Preis, Titel, Ort und Veröffentlichungsdatum und bereitet sie zur weiteren Verarbeitung vor.


•  Details der Implementierung:


•	Header-Konfiguration: Benutzt benutzerdefinierte Header, um Bot-Erkennung zu umgehen.


•	Retries & Timeouts: Implementiert maximal drei Versuche bei Fehlermeldungen und nutzt 10-Sekunden-Timeouts.


•	Extrahierte Datenfelder:


o	Titel

o	Preis

o	Standort

o	Veröffentlichungsdatum


•  Frameworks: BeautifulSoup, requests, Streamlit.¨



•  Fehlerbehandlung: Statuscodes werden überwacht und Fehler (z.B. 404) informativ protokolliert.


### Training

•  Ziel: Das Modell wurde trainiert, um die Wahrscheinlichkeit zu bestimmen, ob ein Motorrad innerhalb von 30 Tagen verkauft wird.


•  Feature Engineering:


•	Tage seit der Veröffentlichung berechnet.


•	Titel analysiert, um Schlüsselbegriffe für Motorradtypen (z. B. "Mountain", "Road") zu extrahieren.


•  Datenbereinigung: Fehlende Preise wurden durch den Medianwert ersetzt; Datumsformate wurden standardisiert.


•  Algorithmus: Random Forest Classifier.


•  Evaluierung: Mittels classification_report geprüft. Metrics wie Precision, Recall und F1-Score wurden berücksichtigt.



### ModelOps Automation

•  Pipeline: Eine automatisierte Pipeline für Scraping, Datenvorbereitung und Modelltraining ist geplant.


•  Abhängigkeiten: Management über requirements.txt. Enthält Bibliotheken wie pandas, scikit-learn, numpy und Flask.


•  Versionierung: Alle Änderungen werden mithilfe von GitHub dokumentiert.


### Deployment

•  Plattform: Azure für Skalierbarkeit und Verfügbarkeit.


•  Interface: Die Anwendung ist über Streamlit zugänglich. Benutzer können Scraping ausführen und Vorhersagen direkt abrufen.


•  Deployment-Prozess: Beinhaltet die Bereitstellung der Anwendung auf Azure mit Integration des Modells.


