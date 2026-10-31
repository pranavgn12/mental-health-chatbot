import {
    PanelLeft,
    Settings,
    ChevronDown
} from "lucide-react";

export default function Navbar({ toggleSidebar }) {

    return (

        <div className="navbar">

            <div className="nav-left">

                <button
                    className="icon-btn"
                    onClick={toggleSidebar}
                >
                    <PanelLeft size={20} />
                </button>

                <h2>AI Companion</h2>

            </div>

            <div className="nav-right">

                <button className="model-btn">

                    gpt-oss-20b

                    <ChevronDown size={16} />

                </button>

                <button
                    className="icon-btn"
                    onClick={() => {

                        localStorage.removeItem("token");
                        localStorage.removeItem("isLoggedIn");

                        window.location.href = "/";

                    }}
                >

                    Logout

                </button>

            </div>

        </div>

    );

}
