import { Link } from "react-router-dom";

interface ExampleWorkflowCardProps {
    title: string;
    description: string;
    to: string;
}

export default function ExampleWorkflowCard({ title, description, to }: ExampleWorkflowCardProps) {
    return (
        <Link className="workflow-card" to={to}>
            <span className="workflow-card__badge">Example</span>
            <h3>{title}</h3>
            <p>{description}</p>
            <span className="workflow-card__cta">Open</span>
        </Link>
    );
}
