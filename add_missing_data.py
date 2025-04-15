import csv
import numpy as np
from scipy.interpolate import interp1d


def extract_columns(data):
    frame_numbers = np.array([int(row['frame_nmr']) for row in data])
    car_ids = np.array([int(float(row['car_id'])) for row in data])
    car_bboxes = np.array([list(map(float, row['car_bbox'][1:-1].split())) for row in data])
    license_plate_bboxes = np.array([list(map(float, row['license_plate_bbox'][1:-1].split())) for row in data])
    return frame_numbers, car_ids, car_bboxes, license_plate_bboxes

def interpolate_missing_frames(prev_frame, curr_frame, prev_bbox, curr_bbox):
    frames_gap = curr_frame - prev_frame
    x = np.array([prev_frame, curr_frame])
    x_new = np.linspace(prev_frame, curr_frame, num=frames_gap, endpoint=False)
    interp_func = interp1d(x, np.vstack((prev_bbox, curr_bbox)), axis=0, kind='linear')
    return interp_func(x_new)[1:]

def create_row(frame_number, car_id, car_bbox, license_plate_bbox, is_imputed, original_row=None):
    row = {
        'frame_nmr': str(frame_number),
        'car_id': str(car_id),
        'car_bbox': ' '.join(map(str, car_bbox)),
        'license_plate_bbox': ' '.join(map(str, license_plate_bbox)),
    }
    if is_imputed:
        row.update({
            'license_plate_bbox_score': '0',
            'license_number': '0',
            'license_number_score': '0',
        })
    else:
        row.update({
            'license_plate_bbox_score': original_row.get('license_plate_bbox_score', '0'),
            'license_number': original_row.get('license_number', '0'),
            'license_number_score': original_row.get('license_number_score', '0'),
        })
    return row

def interpolate_bounding_boxes(data):
    frame_numbers, car_ids, car_bboxes, license_plate_bboxes = extract_columns(data)
    interpolated_data = []
    unique_car_ids = np.unique(car_ids)

    for car_id in unique_car_ids:
        car_mask = car_ids == car_id
        car_frame_numbers = frame_numbers[car_mask]
        car_bboxes_interpolated = []
        license_plate_bboxes_interpolated = []

        for i in range(len(car_bboxes[car_mask])):
            frame_number = car_frame_numbers[i]
            car_bbox = car_bboxes[car_mask][i]
            license_plate_bbox = license_plate_bboxes[car_mask][i]

            if i > 0:
                prev_frame_number = car_frame_numbers[i-1]
                prev_car_bbox = car_bboxes_interpolated[-1]
                prev_license_plate_bbox = license_plate_bboxes_interpolated[-1]

                if frame_number - prev_frame_number > 1:
                    car_bboxes_interpolated.extend(interpolate_missing_frames(
                        prev_frame_number, frame_number, prev_car_bbox, car_bbox
                    ))
                    license_plate_bboxes_interpolated.extend(interpolate_missing_frames(
                        prev_frame_number, frame_number, prev_license_plate_bbox, license_plate_bbox
                    ))

            car_bboxes_interpolated.append(car_bbox)
            license_plate_bboxes_interpolated.append(license_plate_bbox)

        for i, (car_bbox, license_plate_bbox) in enumerate(zip(car_bboxes_interpolated, license_plate_bboxes_interpolated)):
            frame_number = car_frame_numbers[0] + i
            is_imputed = str(frame_number) not in [p['frame_nmr'] for p in data if int(float(p['car_id'])) == car_id]
            original_row = next((p for p in data if int(p['frame_nmr']) == frame_number and int(float(p['car_id'])) == car_id), None)
            interpolated_data.append(create_row(frame_number, car_id, car_bbox, license_plate_bbox, is_imputed, original_row))

    return interpolated_data


# loads CSV file
with open('test.csv', 'r') as file:
    reader = csv.DictReader(file)
    data = list(reader)

# interpolates missing data
interpolated_data = interpolate_bounding_boxes(data)

# writes
header = ['frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox', 'license_plate_bbox_score', 'license_number', 'license_number_score']
with open('test_interpolated.csv', 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=header)
    writer.writeheader()
    writer.writerows(interpolated_data)