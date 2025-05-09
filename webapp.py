from flask import Flask, jsonify, render_template, Response, redirect, url_for, request, flash, session
import cv2
import os
import pickle
import face_recognition
import numpy as np
import cvzone
from datetime import datetime
import json

import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage


app = Flask(__name__)  # initializing
app.secret_key = '12345' # secret key for session management


# database credentials
cred = credentials.Certificate("C:\\Users\\yenzi\\Downloads\\Face-Recognition\\Face-Recognition-System-for-Student-Attendance-main\\serviceAccountKey.json")
firebase_admin.initialize_app(
    cred,
    {
        "databaseURL": "https://todoapp-63646-default-rtdb.firebaseio.com/",
        "storageBucket": "todoapp-63646.firebasestorage.app",
    },
)

bucket = storage.bucket()


def dataset(id,module_id,major):
    # Fetch student information from the database
    studentInfo = db.reference(f"Students/{id}").get()
    attendance = db.reference(f"Attendance/{major}/{module_id}/{id}").get()
    # print(studentInfo)

    # Construct file path
    file_path = f"Face-Recognition-System-for-Student-Attendance-main/static/Files/Images/{id}.png"
    print(f"Looking for file: {file_path}")

    # Fetch the image blob
    blob = bucket.get_blob(file_path)
    if blob is None:
        raise FileNotFoundError(f"Image file for student ID {id} not found in bucket.")

    # Convert the blob to an image
    array = np.frombuffer(blob.download_as_string(), np.uint8)
    imgStudent = cv2.imdecode(array, cv2.COLOR_BGRA2BGR)

    # Calculate the time elapsed since last attendance
    if attendance:
        last_attendance_time = attendance["last_attendance_time"]
        datetimeObject = datetime.strptime(last_attendance_time, "%Y-%m-%d %H:%M:%S")
        secondsElapsed = (datetime.now() - datetimeObject).total_seconds()
    else:
        secondsElapsed = 0  # Since this is the first attendance, no time has elapsed
    # studentInfo["total_attendance"] = attendance["total_attendance"] if attendance else 0
    # studentInfo["last_attendance_time"] = attendance["last_attendance_time"] if attendance else 0


    return studentInfo, imgStudent, secondsElapsed

def studentData(id, module_id, major):
    # Fetch student information from the database
    studentInfo = db.reference(f"Students/{id}").get()
    attendance = db.reference(f"Attendance/{major}/{module_id}/{id}").get()
    if attendance:
        studentInfo["total_attendance"] = attendance.get("total_attendance", 0)
        studentInfo["last_attendance_time"] = attendance.get("last_attendance_time", "N/A")
    else:
        studentInfo["total_attendance"] = 0
        studentInfo["last_attendance_time"] = "N/A"
    
    return studentInfo



already_marked_id_student = []
already_marked_id_admin = []


from flask import request, Response

@app.route("/capture_attendance")
def capture_attendance():    
    major = session.get('major')
    modules_ref = db.reference(f"Modules/{major}")
    modules = modules_ref.get()

    module_id = session.get('module')
    module_name = modules[module_id] if module_id else None
    # module_ref = db.reference(f"Attendance/{major}/{module_id}")
    # attendance = module_ref.get()
    # student_info = []

    # # Check if attendance is not None
    # if attendance:
    #     for student_id, attendance_data in attendance.items():
    #         student_info.append(dataset(student_id, module_id=module_id, major=major))
        
    return render_template("capture_attendance.html", module_name=module_name, id=module_id, numberOfAttendance=0)

@app.route("/video")
def video():
    module_id = session['module']
    major = session.get('major')
    username = session.get('username')  
    return Response(
        generate_frame(module_id,major,username), mimetype="multipart/x-mixed-replace; boundary=frame"
    )

def generate_frame(module_id,major,username):
    # Background and Different Modes

    # video camera
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    imgBackground = cv2.imread("Face-Recognition-System-for-Student-Attendance-main\\static\\Files\\Resources\\background.png")

    folderModePath = "Face-Recognition-System-for-Student-Attendance-main\\static\\Files\\Resources\\Modes\\"
    modePathList = os.listdir(folderModePath)
    imgModeList = []

    for path in modePathList:
        imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))

    modeType = 0
    id = -1
    imgStudent = []
    counter = 0

    # encoding loading ---> to identify if the person is in our database or not.... to detect faces that are known or not

    file = open("EncodeFile.p", "rb")
    encodeListKnownWithIds = pickle.load(file)
    file.close()
    encodedFaceKnown, studentIDs = encodeListKnownWithIds

      # Add a record to the database when the class starts
    class_ref = db.reference(f"Classes/{major}/{module_id}")
    class_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    class_ref.push({
        "module_id": module_id,
        "major": major,
        "startedBy": username,
        "start_time": class_start_time
    })

    while True:
        success, img = capture.read()

        if not success:
            break
        else:
            imgSmall = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

            faceCurrentFrame = face_recognition.face_locations(imgSmall)
            encodeCurrentFrame = face_recognition.face_encodings(
                imgSmall, faceCurrentFrame
            )

            imgBackground[162 : 162 + 480, 55 : 55 + 640] = img
            imgBackground[44 : 44 + 633, 808 : 808 + 414] = imgModeList[modeType]

            if faceCurrentFrame:
                for encodeFace, faceLocation in zip(
                    encodeCurrentFrame, faceCurrentFrame
                ):
                    matches = face_recognition.compare_faces(
                        encodedFaceKnown, encodeFace
                    )
                    faceDistance = face_recognition.face_distance(
                        encodedFaceKnown, encodeFace
                    )

                    matchIndex = np.argmin(faceDistance)

                    y1, x2, y2, x1 = faceLocation
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

                    bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1

                    imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)

                    if matches[matchIndex] == True:
                        id = studentIDs[matchIndex]

                        if counter == 0:
                            cvzone.putTextRect(
                                imgBackground, "Face Detected", (65, 200), thickness=2
                            )
                            cv2.waitKey(1)
                            counter = 1
                            modeType = 1
                    else:
                        cvzone.putTextRect(
                            imgBackground, "Face Detected", (65, 200), thickness=2
                        )
                        cv2.waitKey(3)
                        cvzone.putTextRect(
                            imgBackground, "Face Not Found", (65, 200), thickness=2
                        )
                        modeType = 4
                        counter = 0
                        imgBackground[44 : 44 + 633, 808 : 808 + 414] = imgModeList[
                            modeType
                        ]

                if counter != 0:
                    if counter == 1:
                        studentInfo, imgStudent, secondElapsed = dataset(id=id,module_id=module_id,major=major)
                        print(f"secondsElapsed {secondElapsed}")
                        if int(secondElapsed) > 3600 or secondElapsed == 0:
                            ref = db.reference(f"Students/{id}")
                            att_ref = db.reference(f"Attendance/{major}/{module_id}/{id}")
                            att = att_ref.get()
                            
                            if att:
                                if "total_attendance" in att:
                                    att["total_attendance"] += 1
                                else:
                                    att["total_attendance"] = 1
                                att['last_attendance_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            else:
                                att = {
                                    "total_attendance": 1,
                                    "last_attendance_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                }
                            print(f"att")
                            att_ref.update(att)
                           
                            # ref.child("last_attendance_time").set(
                            #     datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            # )
                        else:
                            modeType = 3
                            counter = 0
                            imgBackground[44 : 44 + 633, 808 : 808 + 414] = imgModeList[
                                modeType
                            ]

                            already_marked_id_student.append(id)
                            already_marked_id_admin.append(id)
                            print(f"student {already_marked_id_student}")
                            print(f"admin {already_marked_id_admin}")

                    if modeType != 3:
                        if 5 < counter <= 10:
                            modeType = 2

                        imgBackground[44 : 44 + 633, 808 : 808 + 414] = imgModeList[
                            modeType
                        ]

                        if counter <= 5:
                            cv2.putText(
                                imgBackground,
                                str(""),
                                (861, 125),
                                cv2.FONT_HERSHEY_COMPLEX,
                                1,
                                (255, 255, 255),
                                1,
                            )
                            cv2.putText(
                                imgBackground,
                                str(studentInfo["major"]),
                                (1006, 550),
                                cv2.FONT_HERSHEY_COMPLEX,
                                0.5,
                                (255, 255, 255),
                                1,
                            )
                            cv2.putText(
                                imgBackground,
                                str(id),
                                (1006, 493),
                                cv2.FONT_HERSHEY_COMPLEX,
                                0.5,
                                (255, 255, 255),
                                1,
                            )
                            cv2.putText(
                                imgBackground,
                                str(""),
                                (910, 625),
                                cv2.FONT_HERSHEY_COMPLEX,
                                0.6,
                                (100, 100, 100),
                                1,
                            )
                            cv2.putText(
                                imgBackground,
                                str(studentInfo["year"]),
                                (1025, 625),
                                cv2.FONT_HERSHEY_COMPLEX,
                                0.6,
                                (100, 100, 100),
                                1,
                            )
                            cv2.putText(
                                imgBackground,
                                str(studentInfo["starting_year"]),
                                (1125, 625),
                                cv2.FONT_HERSHEY_COMPLEX,
                                0.6,
                                (100, 100, 100),
                                1,
                            )

                            (w, h), _ = cv2.getTextSize(
                                str(studentInfo["name"]), cv2.FONT_HERSHEY_COMPLEX, 1, 1
                            )

                            offset = (414 - w) // 2
                            cv2.putText(
                                imgBackground,
                                str(studentInfo["name"]),
                                (808 + offset, 445),
                                cv2.FONT_HERSHEY_COMPLEX,
                                1,
                                (50, 50, 50),
                                1,
                            )

                            imgStudentResize = cv2.resize(imgStudent, (216, 216))

                            imgBackground[
                                175 : 175 + 216, 909 : 909 + 216
                            ] = imgStudentResize

                        counter += 1

                        if counter >= 10:
                            counter = 0
                            modeType = 0
                            studentInfo = []
                            imgStudent = []
                            imgBackground[44 : 44 + 633, 808 : 808 + 414] = imgModeList[
                                modeType
                            ]

            else:
                modeType = 0
                counter = 0

            ret, buffer = cv2.imencode(".jpeg", imgBackground)
            frame = buffer.tobytes()

        yield (b"--frame\r\n" b"Content-Type: image/jpeg \r\n\r\n" + frame + b"\r\n")


#########################################################################################################################


@app.route("/")
def index():
    if session.get('admin_logged_in') != True:
        return redirect(url_for("admin_login"))
    
    major = db.reference(f'Majors/{session["major"]}').get()
    module = db.reference(f'Modules/{session["major"]}/{session["module"]}').get()
    
    return render_template("dashboard.html", fullname=session['fullname'], email=session['email'], username=session['username'], major=major['name'], module=module['name'])




#########################################################################################################################


@app.route("/student_login", methods=["GET", "POST"])
def student_login():
    id = request.form.get("id_number", False)
    email = request.form.get("email", False)
    password = request.form.get("password", False)
    studentIDs, _ = add_image_database()

    if id:
        if id not in studentIDs:
            return render_template(
                "student_login.html", data=" ❌ The id is not registered"
            )
        else:
            secret_key = f"{id}{email}{password}anythingyoulike"
            hash_secret_key = str(hash(secret_key))
            if (
                dataset(id)[0]["password"] == password
                and dataset(id)[0]["email"] == email
            ):
                return redirect(url_for("student", data=id, title=hash_secret_key))
            else:
                id = False
                return render_template(
                    "student_login.html", data=" ❌ Email/Password Incorrect"
                )
    else:
        return render_template("student_login.html")


@app.route("/student/<data>/<title>")
def student(data, title=None):
    studentInfo, imgStudent, secondElapsed = dataset(data)
    hoursElapsed = round((secondElapsed / 3600), 2)

    info = {
        "studentInfo": studentInfo,
        "lastlogin": hoursElapsed,
        "image": imgStudent,
    }
    return render_template("student.html", data=info)


@app.route("/student_attendance_list", methods=["GET", "POST"])
def student_attendance_list():
    major = session['major']
    module = session['module']
    module_ref = db.reference(f"Attendance/{major}/{module}")
    attendance = module_ref.get()
    student_info = []
    
    if attendance:
        for student_id, attendance_data in attendance.items():
            student_info.append(studentData(student_id, module_id=module, major=major))
    
    return render_template("student_attendance_list.html", data=student_info)

#########################################################################################################################


@app.route("/admin_login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        major = request.form.get("major")
        module = request.form.get("module")
        username = request.form.get("username")
        password = request.form.get("password")
        admin_check = request.form.get("adminCheck")

        # Replace '.' with '_' in the username to avoid Firebase path issues
        username_key = username.replace(".", "_")
        user_ref = db.reference(f"Users/{username_key}")
        user = user_ref.get()

        if user:
            if user["password"] == password:
                if admin_check and user.get("usertype") == "admin":
                    session['admin_logged_in'] = True
                    session['major'] = major
                    session['module'] = module
                    session['email'] = user["email"]
                    session['username'] = user["username"]
                    session['fullname'] = user['fullname']
                    return redirect(url_for("admin"))
                elif not admin_check:
                    session['admin_logged_in'] = True
                    session['major'] = major
                    session['module'] = module
                    session['email'] = user["email"]
                    session['username'] = user["username"]
                    session['fullname'] = user['fullname']
                    return redirect(url_for("index"))
                else:
                    flash("❌ You are not authorized to login as admin.")
                    return redirect(url_for("admin_login"))
            else:
                flash("❌ Email/Password Incorrect.")
                return redirect(url_for("admin_login"))
        else:
            flash("❌ The user is not registered")
            return redirect(url_for("admin_login"))
    
    majors_ref = db.reference("Majors")
    majors = majors_ref.get()            

    return render_template("admin_login.html", majors=majors)

@app.route("/get_modules/<major_id>")
def get_modules(major_id):
    modules_ref = db.reference(f"Modules/{major_id}")
    modules = modules_ref.get() or {}
    return jsonify(modules)

@app.route("/logout")
def logout():
    session.pop('admin_logged_in', None)
    session.pop('major', None)
    session.pop('email', None)
    session.pop('username', None)
    return redirect(url_for("admin_login"))

@app.route("/admin")
def admin():
    students_ref = db.reference("Students")
    students = students_ref.get()
    num_students = len(students) if students else 0

    users_ref = db.reference("Users")
    users = users_ref.get()
    num_users = len(users) if users else 0

    majors_ref = db.reference("Majors")
    majors = majors_ref.get()
    num_majors = len(majors) if majors else 0

    data = {
        "num_students": num_students,
        "num_users": num_users,
        "num_majors": num_majors,
    }

    return render_template("admin.html", data=data)


@app.route("/admin/admin_attendance_list", methods=["GET", "POST"])
def admin_attendance_list():
    if request.method == "POST":
        if request.form.get("button_student") == "VALUE1":
            already_marked_id_student.clear()
            return redirect(url_for("admin_attendance_list"))
        else:
            request.form.get("button_admin") == "VALUE2"
            already_marked_id_admin.clear()
            return redirect(url_for("admin_attendance_list"))
    else:
        unique_id_admin = list(set(already_marked_id_admin))
        student_info = []
        for i in unique_id_admin:
            student_info.append(dataset(i))
        return render_template("admin_attendance_list.html", data=student_info)



#########################################################################################################################

def add_image_database():
    folderPath = "Face-Recognition-System-for-Student-Attendance-main/static/Files/Images"
    imgPathList = os.listdir(folderPath)
    imgList = []
    studentIDs = []

    for path in imgPathList:
        imgList.append(cv2.imread(os.path.join(folderPath, path)))
        studentIDs.append(os.path.splitext(path)[0])

        fileName = f"{folderPath}/{path}"
        bucket = storage.bucket()
        blob = bucket.blob(fileName)
        blob.upload_from_filename(fileName)

    return studentIDs, imgList


def findEncodings(images):
    encodeList = []

    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList


@app.route("/admin/add_student", methods=["GET", "POST"])
def add_student():
    id = request.form.get("id", False)
    name = request.form.get("name", False)
    password = request.form.get("password", False)
    dob = request.form.get("dob", False)
    phone = request.form.get("phone", False)
    email = request.form.get("email", False)
    major = request.form.get("major", False)    
    year = request.form.get("year", False)
    starting_year = request.form.get("starting_year", False)
    content = request.form.get("content", False)
    year = int(year)
    starting_year = int(starting_year)

    if request.method == "POST":
        image = request.files["image"]
        filename = f"{'Face-Recognition-System-for-Student-Attendance-main/static/Files/Images'}/{id}.png"
        image.save(os.path.join(filename))

    studentIDs, imgList = add_image_database()

    encodeListKnown = findEncodings(imgList)

    encodeListKnownWithIds = [encodeListKnown, studentIDs]

    file = open("EncodeFile.p", "wb")
    pickle.dump(encodeListKnownWithIds, file)
    file.close()

    if id:
        add_student = db.reference(f"Students")

        add_student.child(id).set(
            {
                "id": id,
                "name": name,
                "password": password,
                "dob": dob,
                "phone": phone,
                "email": email,
                "major": major,
                "starting_year": starting_year,
                "year": year,
                "content": content,
            }
        )
    majors_ref = db.reference("Majors")
    majors = majors_ref.get()

    return render_template("add_student.html", majors=majors)

@app.route("/admin/add_major", methods=["GET", "POST"])
def add_major():
    id = request.form.get("id", False)
    name = request.form.get("name", False)
    created_date = datetime.now().isoformat()

    if id:
        add_major = db.reference(f"Majors")

        add_major.child(id).set(
            {
                "id": id,
                "name": name,
                "created_date": created_date,
            }
        )

    return render_template("add_major.html")

@app.route("/admin/view_majors", methods=["GET", "POST"])
def view_majors():
    if request.method == "POST":
        module_id = request.form.get("module_id")
        module_name = request.form.get("module_name")
        major_id = request.form.get("major_id")

        if major_id and module_id:
            add_module = db.reference(f"Modules/{major_id}")
            add_module.child(module_id).set(
                {
                    "id": module_id,
                    "name": module_name,
                    "created_date": datetime.now().isoformat(),
                }
            )

    majors_ref = db.reference("Majors")
    majors = majors_ref.get()

    # Count the number of modules for each major
    for major_id, major in majors.items():
        modules_ref = db.reference(f"Modules/{major_id}")
        modules = modules_ref.get()
        major['module_count'] = len(modules) if modules else 0

    return render_template("view_majors.html", majors=majors)

@app.route("/admin/view_students", methods=["GET"])
def view_students():
    students_ref = db.reference("Students")
    students = students_ref.get()
    return render_template("view_students.html", students=students)

@app.route("/admin/view_users", methods=["GET"])
def view_users():
    users = db.reference("Users").get()
    return render_template("view_users.html", users=users)

@app.route("/admin/add_user", methods=["GET", "POST"])
def add_user():
    if request.method == "POST":
        id = request.form.get("username")
        name = request.form.get("name")
        password = request.form.get("password")
        phone = request.form.get("phone")
        email = request.form.get("email")
        major = request.form.get("major")
        usertype = request.form.get("usertype")

        # Add user to the database
        add_user_ref = db.reference("Users")
        add_user_ref.child(id).set(
            {
                "username": id,
                "fullname": name,
                "password": password,
                "phonenumber": phone,
                "email": email,
                "department": major,
                "usertype": usertype
            }
        )

    majors_ref = db.reference("Majors")
    majors = majors_ref.get()

    return render_template("add_user.html", majors=majors)




#########################################################################################################################


@app.route("/admin/edit_user", methods=["POST", "GET"])
def edit_user():
    value = request.form.get("edit_student")

    studentInfo, imgStudent, secondElapsed = dataset(value)
    hoursElapsed = round((secondElapsed / 3600), 2)

    info = {
        "studentInfo": studentInfo,
        "lastlogin": hoursElapsed,
        "image": imgStudent,
    }

    return render_template("edit_user.html", data=info)


#########################################################################################################################


@app.route("/admin/save_changes", methods=["POST", "GET"])
def save_changes():
    content = request.get_data()

    dic_data = json.loads(content.decode("utf-8"))

    dic_data = {k: v.strip() for k, v in dic_data.items()}

    dic_data["year"] = int(dic_data["year"])
    dic_data["total_attendance"] = int(dic_data["total_attendance"])
    dic_data["starting_year"] = int(dic_data["starting_year"])

    update_student = db.reference(f"Students")

    update_student.child(dic_data["id"]).update(
        {
            "id": dic_data["id"],
            "name": dic_data["name"],
            "dob": dic_data["dob"],
            "address": dic_data["address"],
            "phone": dic_data["phone"],
            "email": dic_data["email"],
            "major": dic_data["major"],
            "starting_year": dic_data["starting_year"],
            "standing": dic_data["standing"],
            "total_attendance": dic_data["total_attendance"],
            "year": dic_data["year"],
            "last_attendance_time": dic_data["last_attendance_time"],
            "content": dic_data["content"],
        }
    )

    return "Data received successfully!"


#########################################################################################################################


def delete_image(student_id):
    filepath = f"Face-Recognition-System-for-Student-Attendance-main/static/Files/Images/{student_id}.png"

    os.remove(filepath)

    bucket = storage.bucket()
    blob = bucket.blob(filepath)
    blob.delete()

    return "Successful"


@app.route("/admin/delete_user", methods=["POST", "GET"])
def delete_user():
    content = request.get_data()

    student_id = json.loads(content.decode("utf-8"))

    delete_student = db.reference(f"Students")
    delete_student.child(student_id).delete()

    delete_image(student_id)

    studentIDs, imgList = add_image_database()

    encodeListKnown = findEncodings(imgList)

    encodeListKnownWithIds = [encodeListKnown, studentIDs]

    file = open("EncodeFile.p", "wb")
    pickle.dump(encodeListKnownWithIds, file)
    file.close()

    return "Successful"


#########################################################################################################################
if __name__ == "__main__":
    app.run(debug=True)
