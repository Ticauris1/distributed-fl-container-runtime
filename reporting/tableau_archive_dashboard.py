"""Create a local Tableau workbook for archived best-run comparisons."""

from __future__ import annotations

import csv
import os
import re
import sys
import uuid
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_TABLEAU = PROJECT_ROOT / "results" / "archive" / "pre_matched_6encoder_2026-10-04" / "tableau"
SOURCE = ARCHIVE_TABLEAU / "segmentation_experiments.csv"
BEST_SOURCE = ARCHIVE_TABLEAU / "archived_best_encoder_runs.csv"
TABLEAU_WORKBOOK_DIR = Path(
    os.environ.get(
        "TABLEAU_WORKBOOK_DIR",
        str(PROJECT_ROOT / "tableau" / "workbooks"),
    )
).expanduser()
TEMPLATE = TABLEAU_WORKBOOK_DIR / "Book2.twbx"
OUTPUT = TABLEAU_WORKBOOK_DIR / "Archived Segmentation Comparison.twb"
PACKAGED_OUTPUT = OUTPUT.with_suffix(".twbx")
HYPER_OUTPUT = ARCHIVE_TABLEAU / "archived_best_encoder_runs.hyper"
DATASOURCE = "federated.1a26i200exetfh1e6dtjf1ecbj9m"
MATCHED_ENCODERS = {
    "densenet121": "DenseNet121",
    "efficientnet-b3": "EfficientNet-B3",
    "resnet50": "ResNet50",
    "tu-inception_v3": "InceptionV3",
    "tu-mobilenetv2_100": "MobileNetV2",
    "vgg16": "VGG16",
}


def _read(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _best_rows():
    rows = _read(SOURCE)
    best = {}
    for row in rows:
        raw_encoder = row.get("encoder", "")
        if raw_encoder not in MATCHED_ENCODERS:
            continue
        row["encoder"] = MATCHED_ENCODERS[raw_encoder]
        category = "federated" if row.get("training_type") == "federated" else "centralized"
        row["training_type"] = category
        key = (row.get("encoder", ""), category)
        try:
            score = float(row.get("dice") or "-inf")
        except ValueError:
            continue
        if key not in best or score > float(best[key].get("dice") or "-inf"):
            best[key] = row
    selected = sorted(best.values(), key=lambda row: (row.get("encoder", ""), row["training_type"]))
    fields = list(rows[0])
    with BEST_SOURCE.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected)
    return len(selected), len(fields)


def _worksheet(name, measure, caption):
    instance = f"avg:{measure}:qk"
    return f"""    <worksheet name='{name}'>
      <table>
        <view>
          <datasources><datasource caption='Archived best runs' name='{DATASOURCE}' /></datasources>
          <datasource-dependencies datasource='{DATASOURCE}'>
            <column-instance column='[{measure}]' derivation='Avg' name='[{instance}]' pivot='key' type='quantitative' />
            <column caption='{caption}' datatype='real' name='[{measure}]' role='measure' type='quantitative' />
            <column caption='Encoder' datatype='string' name='[encoder]' role='dimension' type='nominal' />
            <column caption='Training Type' datatype='string' name='[training_type]' role='dimension' type='nominal' />
            <column-instance column='[encoder]' derivation='None' name='[none:encoder:nk]' pivot='key' type='nominal' />
            <column-instance column='[training_type]' derivation='None' name='[none:training_type:nk]' pivot='key' type='nominal' />
          </datasource-dependencies>
          <aggregation value='true' />
        </view>
        <style><style-rule element='label'><format attr='display' field='[{DATASOURCE}].[none:training_type:nk]' value='false' /></style-rule></style>
        <panes><pane selection-relaxation-option='selection-relaxation-allow'><view><breakdown value='auto' /></view><mark class='Automatic' /><encodings><color column='[{DATASOURCE}].[none:training_type:nk]' /></encodings></pane></panes>
        <rows>([{DATASOURCE}].[none:encoder:nk] / [{DATASOURCE}].[none:training_type:nk])</rows>
        <cols>[{DATASOURCE}].[{instance}]</cols>
      </table>
      <simple-id uuid='{{{str(uuid.uuid4()).upper()}}}' />
    </worksheet>"""


def _window(name):
    return f"""    <window class='worksheet' maximized='true' name='{name}'>
      <cards><edge name='left'><strip size='300'><card type='pages' /><card type='filters' /><card type='marks' /></strip></edge><edge name='top'><strip size='2147483647'><card type='columns' /></strip><strip size='2147483647'><card type='rows' /></strip><strip size='31'><card type='title' /></strip></edge><edge name='right'><strip size='160'><card pane-specification-id='0' param='[{DATASOURCE}].[none:training_type:nk]' type='color' /></strip></edge></cards>
      <simple-id uuid='{{{str(uuid.uuid4()).upper()}}}' />
    </window>"""


def _create_hyper():
    sys.path.insert(0, "/tmp/tableauhyperapi_pkg")
    from tableauhyperapi import (  # type: ignore
        Connection,
        CreateMode,
        HyperProcess,
        Inserter,
        Nullability,
        SqlType,
        TableDefinition,
        TableName,
        Telemetry,
    )

    rows = _read(BEST_SOURCE)
    type_map = {}
    template_text = TEMPLATE.read_bytes() if TEMPLATE.suffix == ".twb" else b""
    with zipfile.ZipFile(TEMPLATE) as archive:
        twb_name = next(name for name in archive.namelist() if name.endswith(".twb"))
        template_text = archive.read(twb_name)
    for datatype, name in re.findall(rb"<column datatype='([^']+)' name='([^']+)' ordinal='\d+'", template_text):
        type_map.setdefault(name.decode(), datatype.decode())

    sql_types = {
        "string": SqlType.text(),
        "integer": SqlType.big_int(),
        "real": SqlType.double(),
        "boolean": SqlType.bool(),
    }
    fields = list(rows[0])
    columns = [
        TableDefinition.Column(field, sql_types[type_map.get(field, "string")], Nullability.NULLABLE)
        for field in fields
    ]

    def convert(value, datatype):
        if value in (None, ""):
            return None
        if datatype == "integer":
            return int(float(value))
        if datatype == "real":
            return float(value)
        if datatype == "boolean":
            return str(value).strip().lower() in {"true", "1", "yes"}
        return str(value)

    HYPER_OUTPUT.unlink(missing_ok=True)
    table_name = TableName("Extract", "Extract")
    with HyperProcess(Telemetry.DO_NOT_SEND_USAGE_DATA_TO_TABLEAU) as process:
        with Connection(process.endpoint, str(HYPER_OUTPUT), CreateMode.CREATE_AND_REPLACE) as connection:
            connection.catalog.create_schema("Extract")
            connection.catalog.create_table(TableDefinition(table_name, columns))
            with Inserter(connection, table_name) as inserter:
                inserter.add_rows(
                    [[convert(row[field], type_map.get(field, "string")) for field in fields] for row in rows]
                )
                inserter.execute()


def build():
    row_count, column_count = _best_rows()
    with zipfile.ZipFile(TEMPLATE) as archive:
        twb_name = next(name for name in archive.namelist() if name.endswith(".twb"))
        text = archive.read(twb_name).decode("utf-8")

    csv_path = str(BEST_SOURCE)
    text = re.sub(r"csvFile='[^']*'", f"csvFile='{csv_path}'", text)
    text = re.sub(r"directory='[^']*'", f"directory='{BEST_SOURCE.parent}'", text)
    text = re.sub(r"filename='[^']*'", f"filename='{BEST_SOURCE.name}'", text)
    text = re.sub(r"table='\[[^']*#csv\]'", f"table='[{csv_path[:-4]}#csv]'", text)
    text = re.sub(r"gridOrigin='[^']*'", f"gridOrigin='A1:BJ{row_count + 1}:no:A1:BJ{row_count + 1}:1'", text)
    text = text.replace("caption='segmentation_experiments'", "caption='Archived best runs'")
    # Tableau Public requires an extract. Point the template's extract metadata
    # at the real Hyper file that is bundled into the generated .twbx.
    text = re.sub(r"dbname='[^']*\.hyper'", "dbname='Data/archived_best_encoder_runs.hyper'", text)
    text = re.sub(r"\s+updated-database='[^']*'", "", text)

    sheets = [
        ("Best Dice", "dice", "Dice"),
        ("Best IoU", "iou", "Intersection over Union"),
        ("Centralized Test Loss", "test_loss", "Test Loss (not recorded for archived federated runs)"),
        ("Pixel Precision", "pixel_precision", "Pixel Precision"),
        ("Pixel Recall", "pixel_recall", "Pixel Recall"),
    ]
    worksheet_xml = "\n".join(_worksheet(*spec) for spec in sheets)
    window_xml = "\n".join(_window(spec[0]) for spec in sheets)
    text = re.sub(r"  <worksheets>.*?  </worksheets>", f"  <worksheets>\n{worksheet_xml}\n  </worksheets>", text, flags=re.S)
    text = re.sub(r"  <windows[^>]*>.*?  </windows>", f"  <windows source-height='30'>\n{window_xml}\n  </windows>", text, flags=re.S)
    text = re.sub(r"  <thumbnails>.*?  </thumbnails>", "", text, flags=re.S)
    OUTPUT.write_text(text, encoding="utf-8")
    _create_hyper()
    with zipfile.ZipFile(PACKAGED_OUTPUT, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(OUTPUT, OUTPUT.name)
        archive.write(HYPER_OUTPUT, "Data/archived_best_encoder_runs.hyper")
    return row_count


if __name__ == "__main__":
    count = build()
    print(f"[tableau-dashboard] selected_rows={count}")
    print(f"[tableau-dashboard] data={BEST_SOURCE}")
    print(f"[tableau-dashboard] workbook={OUTPUT}")
    print(f"[tableau-dashboard] packaged_workbook={PACKAGED_OUTPUT}")
