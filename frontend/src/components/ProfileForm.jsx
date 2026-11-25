import "../styles/profile.css";

const EMPTY_PROFILE = {
    name: "",
    date_of_birth: "",
    gender: "",
    occupation_status: "",
    degree: "",
    branch: "",
    study_year: "",
    job_title: "",
    city: "",
    state: "",
    country: "",
};

export function emptyProfile() {
    return { ...EMPTY_PROFILE };
}

export function normalizeProfile(profile = {}) {
    return {
        ...EMPTY_PROFILE,
        ...profile,
    };
}

export default function ProfileForm({
    value,
    onChange,
    disabled = false,
    submitLabel,
    onSubmit,
    onCancel,
    showActions = true,
}) {
    const profile = normalizeProfile(value);

    function update(field, nextValue) {
        onChange({ ...profile, [field]: nextValue });
    }

    return (
        <div className="profile-form">
            <div className="profile-form-grid">
                <label className="profile-form-field">
                    <span>Name</span>
                    <input
                        type="text"
                        value={profile.name}
                        onChange={event => update("name", event.target.value)}
                        placeholder="Your name"
                        maxLength={100}
                        disabled={disabled}
                    />
                </label>

                <label className="profile-form-field">
                    <span>Date of birth</span>
                    <input
                        type="date"
                        value={profile.date_of_birth || ""}
                        onChange={event => update("date_of_birth", event.target.value)}
                        disabled={disabled}
                    />
                </label>

                <label className="profile-form-field">
                    <span>Gender</span>
                    <select
                        value={profile.gender}
                        onChange={event => update("gender", event.target.value)}
                        disabled={disabled}
                    >
                        <option value="">Prefer not to say</option>
                        <option value="Female">Female</option>
                        <option value="Male">Male</option>
                        <option value="Non-binary">Non-binary</option>
                        <option value="Other">Other</option>
                    </select>
                </label>

                <label className="profile-form-field">
                    <span>Occupation / Education</span>
                    <select
                        value={profile.occupation_status}
                        onChange={event => {
                            const next = event.target.value;
                            onChange({
                                ...profile,
                                occupation_status: next,
                                ...(next !== "student" ? { degree: "", branch: "", study_year: "" } : {}),
                                ...(next !== "working" ? { job_title: "" } : {}),
                            });
                        }}
                        disabled={disabled}
                    >
                        <option value="">Select one</option>
                        <option value="student">Student</option>
                        <option value="working">Working</option>
                        <option value="other">Other</option>
                    </select>
                </label>

                {profile.occupation_status === "student" && (
                    <>
                        <label className="profile-form-field">
                            <span>Degree</span>
                            <input
                                type="text"
                                value={profile.degree}
                                onChange={event => update("degree", event.target.value)}
                                placeholder="e.g. B.E."
                                maxLength={100}
                                disabled={disabled}
                            />
                        </label>
                        <label className="profile-form-field">
                            <span>Branch</span>
                            <input
                                type="text"
                                value={profile.branch}
                                onChange={event => update("branch", event.target.value)}
                                placeholder="e.g. Computer Science"
                                maxLength={150}
                                disabled={disabled}
                            />
                        </label>
                        <label className="profile-form-field">
                            <span>Current year</span>
                            <select
                                value={profile.study_year}
                                onChange={event => update("study_year", event.target.value)}
                                disabled={disabled}
                            >
                                <option value="">Select year</option>
                                {['1st Year', '2nd Year', '3rd Year', '4th Year', '5th Year', 'Other'].map(year => (
                                    <option key={year} value={year}>{year}</option>
                                ))}
                            </select>
                        </label>
                    </>
                )}

                {(profile.occupation_status === "working" || profile.occupation_status === "other") && (
                    <label className="profile-form-field">
                        <span>{profile.occupation_status === "working" ? "Job title / Occupation" : "Please specify"}</span>
                        <input
                            type="text"
                            value={profile.job_title}
                            onChange={event => update("job_title", event.target.value)}
                            placeholder={profile.occupation_status === "working" ? "e.g. Software Engineer" : "Your occupation or current role"}
                            maxLength={150}
                            disabled={disabled}
                        />
                    </label>
                )}

                <label className="profile-form-field">
                    <span>City</span>
                    <input
                        type="text"
                        value={profile.city}
                        onChange={event => update("city", event.target.value)}
                        placeholder="City"
                        maxLength={100}
                        disabled={disabled}
                    />
                </label>
                <label className="profile-form-field">
                    <span>State</span>
                    <input
                        type="text"
                        value={profile.state}
                        onChange={event => update("state", event.target.value)}
                        placeholder="State"
                        maxLength={100}
                        disabled={disabled}
                    />
                </label>
                <label className="profile-form-field">
                    <span>Country</span>
                    <input
                        type="text"
                        value={profile.country}
                        onChange={event => update("country", event.target.value)}
                        placeholder="Country"
                        maxLength={100}
                        disabled={disabled}
                    />
                </label>
            </div>

            {showActions && onSubmit && (
                <div className="profile-form-actions">
                    {onCancel && (
                        <button type="button" className="profile-form-cancel" onClick={onCancel} disabled={disabled}>
                            Cancel
                        </button>
                    )}
                    <button type="button" className="profile-form-submit" onClick={onSubmit} disabled={disabled}>
                        {submitLabel || "Save profile"}
                    </button>
                </div>
            )}
        </div>
    );
}
