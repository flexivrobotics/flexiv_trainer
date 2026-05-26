import { NavLink, Route, Routes } from "react-router-dom";

import HomePage from "./routes/HomePage";
import TeleoperationPage from "./routes/TeleoperationPage";
import TrainingPage from "./routes/TrainingPage";

const navItems = [
    { to: "/teleoperation", label: "Teleoperation" },
    { to: "/training", label: "Training" },
];

export default function App() {
    return (
        <div className="app-shell">
            <header className="app-nav">
                <NavLink className="brand" to="/">
                    Flexiv Trainer
                </NavLink>
                <nav>
                    {navItems.map((item) => (
                        <NavLink className="nav-link" key={item.to} to={item.to}>
                            {item.label}
                        </NavLink>
                    ))}
                </nav>
            </header>

            <main>
                <Routes>
                    <Route element={<HomePage />} path="/" />
                    <Route element={<TeleoperationPage />} path="/teleoperation" />
                    <Route element={<TrainingPage />} path="/training" />
                </Routes>
            </main>
        </div>
    );
}
